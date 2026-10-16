"""Shared helpers for the shots/hardware lab (does not modify any repo file).

Shot-based QAOA training: every training objective evaluation is estimated from N measurement
shots drawn (seeded multinomial) from the exact output distribution of the current circuit,
exactly what a hardware run with N shots per evaluation would return. The objective itself is
the same QUBO-only Gibbs objective as the main pipeline,

    F_hat(theta) = -log( (1/N) sum_{s=1..N} exp(-eta (E(x_s) - E_min) / sigma_E) ),  x_s ~ |<x|psi(theta)>|^2,

so it never sees the official cost. All evaluations (ramp probes, embedding check, optimiser
calls, re-evaluations used to pick the final parameters) count toward the 300-evaluation cap.
"""
import os
import sys
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import baselines as bl  # noqa: E402
import circuit as cq  # noqa: E402
import qaoa  # noqa: E402
import qubo as qb  # noqa: E402
import run  # noqa: E402
import traffic_model as tm  # noqa: E402

SHOTS = 4096
MAX_EVALS = 300


def val_set():
    """Common validation set: scenario R + 60 scenarios from generator seed 555."""
    return [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(60, seed=555)


def tune_set(n=20):
    """Tuning set (never VAL): generator seed 31."""
    return tm.random_scenarios(n, seed=31)


# ----------------------------------------------------------------------------- shot-based objective
def shot_objective(ansatz, energies, n_shots, rng, kind='gibbs', eta=2.0):
    """N-shot estimator of the training objective (QUBO energies only)."""
    if kind == 'gibbs':
        w = np.exp(-eta * (energies - energies.min()) / (energies.std() or 1.0))
    elif kind == 'energy':
        w = energies
    else:
        raise ValueError(kind)

    probs = ansatz if callable(ansatz) else ansatz.probabilities   # ansatz or params -> probabilities

    def f(params):
        pr = probs(params)
        cnt = rng.multinomial(n_shots, pr)
        m = float(cnt @ w) / n_shots
        return -np.log(max(m, 1e-300)) if kind == 'gibbs' else m
    return f


class _Budget(Exception):
    pass


def train_shots(h, J, energies, n_shots, seed, p=4, cost_mode='split', mixer_mode='qubit', kind='gibbs', eta=2.0,
                max_evals=MAX_EVALS, ramp_dts=(0.75, 1.0, 1.25), stage1_evals=40, method='cobyla',
                rhobeg=0.15, rho2=None, tol=1e-7, restarts=True, reeval=0, reeval_k=5, select='best',
                spsa_a=0.1, spsa_c=0.1, spsa_A=10, spsa_avg=0.25, spsa_mom=0.0, spsa_clip=None, prob_fn=None):
    """Shot-noise training, same two-stage structure as qaoa.train.

    method   'cobyla'  : COBYLA in both stages; if COBYLA stops early (trust radius reached tol) and budget
                         remains, it is restarted from the stage incumbent with half the initial radius
             'spsa'    : COBYLA in stage 1 (standard ansatz), SPSA in stage 2 (rich ansatz, 2 evals / iteration)
    reeval   number of evaluations held back at the end; they re-measure the `reeval_k` best distinct
             candidates (round robin) and the final parameters are the candidate with the best MEAN
             estimate (mitigates the winner's curse of picking the minimum of noisy values).
    select   'best' : final parameters = candidate with the lowest single noisy estimate (winner's curse)
             'last' : (spsa) final parameters = SPSA iterate average (no selection on noisy values)
    prob_fn  optional map (final-ansatz params -> outcome probabilities), e.g. a noisy routed device;
             default = the exact ideal circuit distribution.
    Returns (ansatz, params, history) with len(history) <= max_evals (asserted).
    """
    rng = np.random.default_rng([seed, 7919])
    final = qaoa.Ansatz(h, J, p, cost_mode, mixer_mode)
    base = qaoa.Ansatz(h, J, p, 'single', 'single')
    rich = (cost_mode, mixer_mode) != ('single', 'single')
    p_final = prob_fn if prob_fn is not None else final
    p_base = (lambda x: prob_fn(final.lift_params(x, base))) if prob_fn is not None else base
    rho2 = rhobeg / 2 if rho2 is None else rho2
    history = []
    cands = []          # [value, params in the returned parameterisation, all estimates]
    budget = {'cap': max_evals - reeval}

    def counted(fun, a, stage):
        def g(x):
            if len(history) >= budget['cap']:
                raise _Budget
            v = fun(x)
            history.append(v)
            xf = final.lift_params(x, base) if (a is base and rich) else np.array(x, float)
            cands.append([v, xf, [v]])
            stage.append((v, np.array(x, float)))
            return v
        return g

    def cobyla(fun, x0, rho, stage):
        while len(history) < budget['cap']:
            n0 = len(history)
            try:
                minimize(fun, x0, method='COBYLA',
                         options={'maxiter': budget['cap'] - len(history), 'rhobeg': rho, 'tol': tol})
            except _Budget:
                return
            if not restarts or len(history) - n0 < 5:
                return
            x0, rho = min(stage, key=lambda c: c[0])[1], rho / 2

    s1 = []
    fb = counted(shot_objective(p_base, energies, n_shots, rng, kind, eta), base, s1)
    for dt in ramp_dts:
        fb(base.ramp(dt))
    x_start = min(s1, key=lambda c: c[0])[1]
    if not rich:
        cobyla(fb, x_start, rhobeg, s1)
    else:
        budget['cap'] = stage1_evals
        cobyla(fb, x_start, rhobeg, s1)
        budget['cap'] = max_evals - reeval
        x0 = final.lift_params(min(s1, key=lambda c: c[0])[1], base)
        s2 = []
        ff = counted(shot_objective(p_final, energies, n_shots, rng, kind, eta), final, s2)
        if method == 'cobyla':
            cobyla(ff, x0, rho2, s2)
        elif method == 'spsa':
            x_last = _spsa(ff, x0, rng, budget['cap'] - len(history), spsa_a, spsa_c, spsa_A, spsa_avg, spsa_mom, spsa_clip)
        else:
            raise ValueError(method)
    # ---- final selection
    budget['cap'] = max_evals
    if reeval > 0:
        uniq = []
        for c in sorted(cands, key=lambda c: c[0]):
            if not any(np.allclose(c[1], u[1]) for u in uniq):
                uniq.append(c)
            if len(uniq) >= reeval_k:
                break
        fr = shot_objective(p_final if rich else p_base, energies, n_shots, rng, kind, eta)
        for i in range(reeval):
            if len(history) >= max_evals:
                break
            u = uniq[i % len(uniq)]
            v = fr(u[1])
            history.append(v)
            u[2].append(v)
        x = min(uniq, key=lambda u: np.mean(u[2]))[1]
    elif select == 'last' and method == 'spsa' and rich:
        x = x_last          # SPSA iterate average: no selection on noisy values at all
    else:
        x = min(cands, key=lambda c: c[0])[1]
    assert len(history) <= max_evals
    return (final if rich else base), np.array(x, float), history


def _spsa(f, x0, rng, evals, a, c, A, avg_frac, mom=0.0, clip=None):
    """Plain SPSA (Spall 1998 gains alpha=0.602, gamma=0.101), Rademacher perturbations, 2 evals / iteration.
    The final iterate average (last avg_frac of iterates) is evaluated once and enters the candidate pool."""
    x = np.array(x0, float)
    m = np.zeros_like(x)
    iters = max((evals - 1) // 2, 0)
    tail = []
    for k in range(iters):
        ck = c / (k + 1) ** 0.101
        ak = a / (k + 1 + A) ** 0.602
        d = rng.choice([-1.0, 1.0], size=x.size)
        try:
            fp = f(x + ck * d)
            fm = f(x - ck * d)
        except _Budget:
            break
        m = mom * m + (1 - mom) * (fp - fm) / (2 * ck) * d   # mom = 0: plain SPSA
        step = ak * m
        if clip is not None:   # trust-region-like cap on every coordinate of the update
            step = np.clip(step, -clip, clip)
        x = x - step
        if k >= iters * (1 - avg_frac):
            tail.append(x.copy())
    xa = np.mean(tail, axis=0) if tail else x
    try:
        f(xa)
    except _Budget:
        pass
    return xa


# ----------------------------------------------------------------------------- full solve with shot training
def solve_shots(s, seed, n_shots, tcfg=None, shots=SHOTS, final_sample=True):
    """Mirror of run.solve with the exact trainer replaced by train_shots. Returns metrics dict."""
    cfg = run.DEFAULTS
    tcfg = tcfg or {}
    offset, Q = qb.build_qubo(s, measure=cfg['measure'], kappa=cfg['kappa'])
    h, J, _ = qb.qubo_to_ising(Q, offset)
    e = qb.energies(Q, offset)
    if n_shots is None:   # exact training (reference path, identical to run.solve)
        ansatz, x, hist = qaoa.train(h, J, e, p=tcfg.get('p', cfg['p']), cost_mode=cfg['cost_mode'],
                                     mixer_mode=cfg['mixer_mode'], kind=cfg['kind'], eta=cfg['eta'],
                                     stage1_evals=cfg['stage1_evals'], ramp_dts=cfg['ramp_dts'])
    else:
        kw = dict(p=cfg['p'], cost_mode=cfg['cost_mode'], mixer_mode=cfg['mixer_mode'], kind=cfg['kind'],
                  eta=cfg['eta'], stage1_evals=cfg['stage1_evals'], ramp_dts=cfg['ramp_dts'])
        kw.update(tcfg)
        ansatz, x, hist = train_shots(h, J, e, n_shots, seed, **kw)
    assert len(hist) <= MAX_EVALS
    costs = tm.cost_table(s)
    pr = ansatz.probabilities(x)
    out = {'scenario': s.get('name'), 'seed': seed, 'n_train_shots': n_shots, 'evals': len(hist),
           'exact': bl.distribution_metrics(costs, pr)}
    if final_sample:
        qc = cq.build_circuit(ansatz, x)
        counts = cq.sample(qc, shots, seed)
        m = bl.sample_metrics(costs, counts)
        out['sampled'] = {k: m[k] for k in ('E_Q', 'lift', 'enrichment', 'p_optimal')}
    return out


def summarize(rows, key='sampled'):
    L = np.array([r[key]['lift'] for r in rows])
    En = np.array([r[key]['enrichment'] for r in rows])
    return {'mean_lift': float(L.mean()), 'min_lift': float(L.min()), 'mean_enrichment': float(En.mean()),
            'min_enrichment': float(En.min()), 'n': len(rows)}
