"""Shared harness for the parameter-transfer initialisation lab (lab/transfer_init).

Nothing in the repo is modified: repo modules are imported from the repo root.

Scenario sets
  LEARN(n) = traffic_model.random_scenarios(n, seed=31)   offline learning of the transfer angles
  DEV(n)   = traffic_model.random_scenarios(n, seed=7)    selection of run-time strategy / budget split
  VAL      = [scenario R] + traffic_model.random_scenarios(60, seed=555)   reporting only
Seeds 2026, 99 and 555 are never used for learning or selection.

Contents
  * BatchQAOA  - batched numpy statevector of qaoa.Ansatz (same unitary, same parameter layout) for many
                 scenarios at once, with an exact adjoint (reverse-mode) gradient of the Gibbs / energy
                 objectives. Used only OFFLINE to learn the transfer angles.
  * train_ti   - run-time budgeted trainer = qaoa.train plus transfer starts and budget-split options.
                 Every objective evaluation is counted (<= 300, asserted); objective = QUBO energies only.
  * solve_ti   - run.solve-equivalent pipeline (QUBO -> train -> Qiskit circuit -> 4096 shots, seed) that
                 returns the judging metrics; with cfg=None-like settings it reproduces run.solve exactly.
"""
import os
import sys
import pickle
import time
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import baselines as bl  # noqa: E402
import circuit as cq  # noqa: E402
import qaoa  # noqa: E402
import qubo as qb  # noqa: E402
import traffic_model as tm  # noqa: E402

N, DIM = 8, 256
CACHE = HERE / 'cache'
CACHE.mkdir(exist_ok=True)
FORBIDDEN_SEEDS = (2026, 99, 555)


# ----------------------------------------------------------------------------- scenario sets
class Problem:
    """Scenario + official cost table (benchmark only) + QUBO / Ising (training only)."""

    def __init__(self, s, costs, measure='adaptive', kappa=1.0):
        self.s, self.costs = s, np.asarray(costs, float)
        self.name = s.get('name')
        self.offset, self.Q = qb.build_qubo(s, measure=measure, kappa=kappa)
        self.h, self.J, self.const = qb.qubo_to_ising(self.Q, self.offset)
        self.E = qb.energies(self.Q, self.offset)


def _scen_set(key, maker):
    f = CACHE / f'{key}.pkl'
    if f.exists():
        return pickle.loads(f.read_bytes())
    data = [(s, tm.cost_table(s)) for s in maker()]
    f.write_bytes(pickle.dumps(data))
    return data


def scen_set(n, seed):
    assert seed not in FORBIDDEN_SEEDS, 'reserved seed'
    return _scen_set(f's{seed}_n{n}', lambda: tm.random_scenarios(n, seed=seed))


def learn_set(n=150):
    return scen_set(n, 31)


def dev_set(n=60):
    return scen_set(n, 7)


def val_set():
    def mk():
        R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
        return [R] + tm.random_scenarios(60, seed=555)
    return _scen_set('val_R_s555_n60', mk)


def problems(data, measure='adaptive', kappa=1.0):
    return [Problem(s, c, measure, kappa) for s, c in data]


# ----------------------------------------------------------------------------- batched simulator (offline)
def gibbs_weights(E, eta):
    return np.exp(-eta * (E - E.min()) / (E.std() or 1.0))


class BatchQAOA:
    """Same ansatz as qaoa.Ansatz(h, J, p, cost_mode, mixer_mode) for a batch of scenarios.

    Parameter layout identical to qaoa.Ansatz: [g (p x mg) row-major, b (p x mb) row-major].
    Objective (per scenario): kind='gibbs' -> -log sum_x P(x) w(x), w = exp(-eta (E - Emin)/std E)
                              kind='energy' -> sum_x P(x) (E - mean E)/std E
    Returned objective = weighted mean over scenarios (weights default uniform).
    """

    def __init__(self, probs, p=4, cost_mode='split', mixer_mode='qubit', kind='gibbs', eta=2.0):
        self.p, self.cost_mode, self.mixer_mode, self.kind = p, cost_mode, mixer_mode, kind
        self.ans = [qaoa.Ansatz(pr.h, pr.J, p, cost_mode, mixer_mode) for pr in probs]
        self.T = np.stack([a.T for a in self.ans])            # (S, 256, mg), already divided by scale
        self.mg, self.mb = self.ans[0].mg, self.ans[0].mb
        self.n_params = self.ans[0].n_params
        if kind == 'gibbs':
            self.W = np.stack([gibbs_weights(pr.E, eta) for pr in probs])          # (S, 256)
        elif kind == 'energy':
            self.W = np.stack([(pr.E - pr.E.mean()) / (pr.E.std() or 1.0) for pr in probs])
        else:
            raise ValueError(kind)
        self.S = len(probs)
        self.weights = np.full(self.S, 1.0 / self.S)

    def split(self, x):
        x = np.asarray(x, float)
        return x[:self.p * self.mg].reshape(self.p, self.mg), x[self.p * self.mg:].reshape(self.p, self.mb)

    @staticmethod
    def _rx(psi, ax, beta):
        """exp(+i beta X) on schedule bit `ax` (tensor axis ax of the (2,)*8 reshape), batched."""
        S = psi.shape[0]
        t = psi.reshape(S, 2 ** ax, 2, 2 ** (N - 1 - ax))
        c, s = np.cos(beta), 1j * np.sin(beta)
        t0, t1 = t[:, :, 0, :], t[:, :, 1, :]
        out = np.empty_like(t)
        out[:, :, 0, :] = c * t0 + s * t1
        out[:, :, 1, :] = s * t0 + c * t1
        return out.reshape(S, DIM)

    @staticmethod
    def _x(psi, ax):
        S = psi.shape[0]
        return psi.reshape(S, 2 ** ax, 2, 2 ** (N - 1 - ax))[:, :, ::-1, :].reshape(S, DIM)

    def states(self, x):
        g, b = self.split(x)
        psi = np.full((self.S, DIM), 1 / np.sqrt(DIM), dtype=complex)
        for k in range(self.p):
            psi = psi * np.exp(-1j * (self.T @ g[k]))
            bk = np.broadcast_to(b[k], (N,))
            for ax in range(N):
                psi = self._rx(psi, ax, bk[ax])
        return psi

    def probabilities(self, x):
        pr = np.abs(self.states(x)) ** 2
        return pr / pr.sum(1, keepdims=True)

    def per_scenario(self, x):
        pr = self.probabilities(x)
        A = (pr * self.W).sum(1)
        return -np.log(A) if self.kind == 'gibbs' else A

    def value(self, x):
        return float(self.weights @ self.per_scenario(x))

    def value_grad(self, x):
        """Objective and exact gradient (adjoint method, one forward + one backward sweep)."""
        g, b = self.split(x)
        psi = self.states(x)
        pr = np.abs(psi) ** 2
        A = (pr * self.W).sum(1)
        if self.kind == 'gibbs':
            vals = -np.log(A)
            lam = -(self.weights / A)[:, None] * self.W * psi          # dF/dpsi*
        else:
            vals = A
            lam = self.weights[:, None] * self.W * psi
        gg, gb = np.zeros_like(g), np.zeros((self.p, N))
        for k in reversed(range(self.p)):
            bk = np.broadcast_to(b[k], (N,))
            for ax in reversed(range(N)):
                # psi, lam are after RX_ax ; d/dbeta: generator i X
                gb[k, ax] = 2 * np.real(np.sum(np.conj(lam) * (1j * self._x(psi, ax))))
                psi = self._rx(psi, ax, -bk[ax])
                lam = self._rx(lam, ax, -bk[ax])
            # phase layer: psi' = exp(-i T g) psi ; d/dg_j: generator -i T_j
            for j in range(self.mg):
                gg[k, j] = 2 * np.real(np.sum(np.conj(lam) * (-1j * self.T[:, :, j] * psi)))
            ph = np.exp(1j * (self.T @ g[k]))
            psi, lam = psi * ph, lam * ph
        gb = gb.sum(1, keepdims=True) if self.mb == 1 else gb
        return float(self.weights @ vals), np.concatenate([gg.ravel(), gb.ravel()])


def lbfgs(model, x0, maxiter=500, gtol=1e-7):
    r = minimize(model.value_grad, np.asarray(x0, float), jac=True, method='L-BFGS-B',
                 options={'maxiter': maxiter, 'gtol': gtol})
    return r.x, float(r.fun)


def ramp(p, mg, mb, dt):
    k = (np.arange(p) + 0.5) / p
    return np.concatenate([np.repeat((k * dt)[:, None], mg, 1).ravel(), np.repeat(((1 - k) * dt)[:, None], mb, 1).ravel()])


def lift_vec(x, p, src, dst):
    """Embed params of ansatz (cost_mode, mixer_mode)=src into dst (same p); identical state."""
    a_src = qaoa.Ansatz(np.zeros(N), np.zeros((N, N)), p, *src)
    a_dst = qaoa.Ansatz(np.zeros(N), np.zeros((N, N)), p, *dst)
    return a_dst.lift_params(np.asarray(x, float), a_src)


# ----------------------------------------------------------------------------- run-time trainer
class BudgetExceeded(Exception):
    pass


def train_ti(h, J, energies, p=4, cost_mode='split', mixer_mode='qubit', kind='gibbs', eta=2.0, alpha=0.25,
             max_evals=300, init='ramp', ramp_dts=(0.75, 1.0, 1.25), transfer=None, transfer_base=None,
             stage1_evals=40, rhobeg=0.15, rhobeg2=None, init_seed=0, n_random=3, cache=True,
             scale_evals=0, scale_mode='global3', scale_rho=0.1):
    """Budgeted training with optional parameter-transfer starts.

    init:
      'ramp'      - current pipeline: 3 linear-ramp probes on the standard ansatz, COBYLA (standard) to
                    stage1_evals, lift into the rich ansatz, COBYLA (rich) to max_evals.
      'random'    - as 'ramp' but n_random uniform[0, pi/2) probes (ablation).
      'transfer'  - evaluate the fixed transfer vector of the RICH ansatz (1 evaluation) and run COBYLA on the
                    rich ansatz from it with the remaining budget (stage 1 skipped).
      'transfer+ramp' - probes = ramp probes (standard ansatz) + transfer vector (rich). If the transfer probe
                    wins, skip stage 1; otherwise proceed as 'ramp'.
      'transfer_base' - transfer vector for the STANDARD ansatz replaces the ramp probes; stage 1 + stage 2 as 'ramp'.
    cache: a point already evaluated in the same parameterisation is not re-evaluated (COBYLA's first call
           at its x0); this saves an evaluation, never adds one.
    rhobeg2: COBYLA rhobeg for the rich stage (default rhobeg/2 as in qaoa.train).
    scale_evals: (transfer start only) budget of a low-dimensional COBYLA stage over multiplicative factors
           of the transfer angles before the full rich stage: 'global3' = (field gamma, coupling gamma, beta)
           factors; 'layer12' = the same three factors per layer. Requires the split-cost layout (last cost column
           = coupling angle), true for cost_mode 'split' and 'field'.
    Every evaluation counts toward max_evals (asserted). Objective uses QUBO energies only.
    """
    final = qaoa.Ansatz(h, J, p, cost_mode, mixer_mode)
    base = qaoa.Ansatz(h, J, p, 'single', 'single')
    rich = (cost_mode, mixer_mode) != ('single', 'single')
    rhobeg2 = rhobeg / 2 if rhobeg2 is None else rhobeg2
    history = []
    best = {'x': None, 'v': np.inf, 'ansatz': None}
    memo = {}

    def counted(f, a, tag):
        def g(x):
            x = np.asarray(x, float)
            key = (tag, x.tobytes())
            if cache and key in memo:
                return memo[key]
            if len(history) >= max_evals:
                raise BudgetExceeded
            v = f(x)
            history.append(v)
            memo[key] = v
            if v < best['v']:
                best.update(x=x.copy(), v=v, ansatz=a)
            return v
        return g

    def cobyla(f, x0, evals, rho, tag):
        # COBYLA's first call is at x0; if x0 is cached it is free, so give COBYLA one more call.
        if evals <= 0 or len(history) >= max_evals:
            return
        evals += int(cache and (tag, np.asarray(x0, float).tobytes()) in memo)
        try:
            minimize(f, x0, method='COBYLA', options={'maxiter': evals, 'rhobeg': rho, 'tol': 1e-7})
        except BudgetExceeded:
            pass

    fb = counted(qaoa.make_objective(base, energies, kind, alpha, eta), base, 'base')
    ff = counted(qaoa.make_objective(final, energies, kind, alpha, eta), final, 'final')

    if init in ('ramp', 'transfer+ramp'):
        for dt in ramp_dts:
            fb(base.ramp(dt))
    elif init == 'random':
        rng = np.random.default_rng(init_seed)
        for _ in range(n_random):
            fb(rng.uniform(0, np.pi / 2, base.n_params))
    elif init == 'transfer_base':
        fb(np.asarray(transfer_base, float))
    elif init != 'transfer':
        raise ValueError(init)
    if init in ('transfer', 'transfer+ramp'):
        ff(np.asarray(transfer, float))

    if not rich:
        cobyla(fb, best['x'], max_evals - len(history), rhobeg, 'base')
        return base, best['x'], history

    if best['ansatz'] is final:            # transfer start won: no standard-ansatz stage
        if scale_evals > 0:                # optional low-dimensional stage: rescale the transfer angles
            x_t = best['x'].copy()
            g_t, b_t = final.split(x_t)
            n_g = 1 if scale_mode == 'global3' else p

            def scaled(sv):
                sv = np.asarray(sv, float)
                a, c, d = sv[:n_g], sv[n_g:2 * n_g], sv[2 * n_g:]
                G = g_t.copy()
                G[:, :-1] *= a[:, None] if n_g > 1 else a[0]
                G[:, -1] *= c
                B = b_t * (d[:, None] if n_g > 1 else d[0])
                return final.join(G, B)

            fs = lambda sv: ff(scaled(sv))
            if len(history) < max_evals:
                try:
                    minimize(fs, np.ones(3 * n_g), method='COBYLA',
                             options={'maxiter': scale_evals + 1, 'rhobeg': scale_rho, 'tol': 1e-7})
                except BudgetExceeded:
                    pass
        cobyla(ff, best['x'], max_evals - len(history), rhobeg2, 'final')
    else:
        cobyla(fb, best['x'], stage1_evals - len(history), rhobeg, 'base')
        x0 = final.lift_params(best['x'], base)
        if not cache:
            ff(x0)
        cobyla(ff, x0, max_evals - len(history), rhobeg2, 'final')
    if best['ansatz'] is not final:
        best['x'] = final.lift_params(best['x'], base)
    assert len(history) <= max_evals
    return final, best['x'], history


# ----------------------------------------------------------------------------- full pipeline
def solve_ti(s, seed, cfg, shots=4096, costs=None, sample=True, measure='adaptive', kappa=1.0):
    """QUBO -> train_ti -> Qiskit circuit -> `shots` samples (StatevectorSampler(seed)) -> judging metrics.

    Same downstream steps as run.solve. Returns dict with sampled metrics ('sampled'), exact-distribution
    metrics ('exact'), number of evaluations, surrogate regret and timing.
    """
    t0 = time.time()
    offset, Q = qb.build_qubo(s, measure=measure, kappa=kappa)
    h, J, _ = qb.qubo_to_ising(Q, offset)
    E = qb.energies(Q, offset)
    ansatz, params, hist = train_ti(h, J, E, **cfg)
    assert len(hist) <= 300
    costs = tm.cost_table(s) if costs is None else np.asarray(costs, float)
    probs = ansatz.probabilities(params)
    out = {'name': s.get('name'), 'evals': len(hist), 'final_objective': float(min(hist)),
           'exact': bl.distribution_metrics(costs, probs),
           'regret_rel': float((costs[int(np.argmin(E))] - costs.min()) / costs.min()) if costs.min() > 0 else 0.0,
           'params': params.tolist()}
    if sample:
        qc = cq.build_circuit(ansatz, params)
        pc = cq.circuit_probabilities(qc)
        assert np.abs(pc - probs).max() < 1e-8
        counts = cq.sample(qc, shots, seed)
        m = bl.sample_metrics(costs, counts)
        out['sampled'] = {k: m[k] for k in ('E_Q', 'lift', 'enrichment', 'p_optimal', 'C_star', 'C_rand')}
    out['seconds'] = time.time() - t0
    return out


def summarize(rows, which='sampled'):
    L = np.array([r[which]['lift'] for r in rows])
    En = np.array([r[which]['enrichment'] for r in rows])
    P = np.array([r[which]['p_optimal'] for r in rows])
    return {'n': len(rows), 'mean_lift': float(L.mean()), 'median_lift': float(np.median(L)), 'min_lift': float(L.min()),
            'mean_enrichment': float(En.mean()), 'min_enrichment': float(En.min()), 'mean_p_opt': float(P.mean())}


def fmt(sm):
    return (f"lift {sm['mean_lift']:.4f} (min {sm['min_lift']:.3f}) enr {sm['mean_enrichment']:.2f} "
            f"(min {sm['min_enrichment']:.2f}) P(opt) {sm['mean_p_opt']:.3f}")
