"""QAOA for the traffic QUBO.

Qubit mapping: schedule bit i (A1..A4 = 0..3, B1..B4 = 4..7) lives on qubit 7 - i and is
measured into clbit 7 - i. Qiskit prints clbits as c7..c0, so every Qiskit count key reads
directly as the schedule string A1 A2 A3 A4 B1 B2 B3 B4, and the basis-state index is
int(schedule, 2).

Ansatz (p layers), H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j (Ising form of the QUBO,
divided by its standard deviation so angles are scenario-independent):

    |psi> = prod_k  U_M(beta_k) U_C(gamma_k)  |+>^8
    U_C   = exp(-i [ sum_i g_{k,i} h_i Z_i + g_{k,J} sum J_ij Z_i Z_j ])
    U_M   = prod_i exp(+i b_{k,i} X_i)          (= RX(-2 b_{k,i}); H_M = -sum X)

cost_mode  'single' : g_{k,i} = g_{k,J} = gamma_k              (standard QAOA)
           'split'  : one angle for all fields, one for all couplings
           'field'  : one angle per field term, one for all couplings
mixer_mode 'single' : b_{k,i} = beta_k                          (standard QAOA)
           'qubit'  : one angle per qubit (multi-angle QAOA, Herrman et al. 2022)

Training uses an exact statevector simulator written in numpy (identical unitary to the
Qiskit circuit; verified in tests). Each call of the objective is one "training objective
evaluation"; the total is hard-capped (default 300).
"""
import numpy as np
from scipy.optimize import minimize

import qubo as qb

N = qb.N
DIM = 2 ** N
Z = 1 - 2 * qb.X  # Z[s, i] = eigenvalue of Z on schedule bit i in basis state s

COST_WIDTH = {'single': 1, 'split': 2, 'field': N + 1}
MIXER_WIDTH = {'single': 1, 'qubit': N}


class BudgetExceeded(Exception):
    pass


class Ansatz:
    def __init__(self, h, J, p=4, cost_mode='single', mixer_mode='single'):
        self.h, self.J, self.p = np.asarray(h, float), np.asarray(J, float), p
        self.cost_mode, self.mixer_mode = cost_mode, mixer_mode
        field = Z * self.h                       # (256, 8): h_i z_i
        coup = np.einsum('si,ij,sj->s', Z, self.J, Z)
        self.diag = field.sum(1) + coup          # = E_qubo - const
        self.scale = float(self.diag.std()) or 1.0  # angles act on H_C / scale (guard: flat H)
        if cost_mode == 'single':
            T = self.diag[:, None]
        elif cost_mode == 'split':
            T = np.column_stack([field.sum(1), coup])
        else:
            T = np.column_stack([field, coup])
        self.T = T / self.scale
        self.mg, self.mb = COST_WIDTH[cost_mode], MIXER_WIDTH[mixer_mode]

    @property
    def n_params(self):
        return self.p * (self.mg + self.mb)

    def split(self, params):
        params = np.asarray(params, float)
        g = params[:self.p * self.mg].reshape(self.p, self.mg)
        b = params[self.p * self.mg:].reshape(self.p, self.mb)
        return g, b

    def join(self, g, b):
        return np.concatenate([np.asarray(g, float).ravel(), np.asarray(b, float).ravel()])

    def layer_angles(self, params):
        """Physical angles per layer: (field angles (8,), coupling angle, mixer angles (8,))."""
        g, b = self.split(params)
        out = []
        for k in range(self.p):
            if self.cost_mode == 'single':
                gf, gj = np.full(N, g[k, 0]), g[k, 0]
            elif self.cost_mode == 'split':
                gf, gj = np.full(N, g[k, 0]), g[k, 1]
            else:
                gf, gj = g[k, :N], g[k, N]
            out.append((gf / self.scale, gj / self.scale, np.broadcast_to(b[k], (N,)).copy()))
        return out

    def statevector(self, params):
        g, b = self.split(params)
        psi = np.full(DIM, 1 / np.sqrt(DIM), dtype=complex)
        for k in range(self.p):
            phase = self.T @ g[k] if self.mg > 1 else self.T[:, 0] * g[k, 0]
            psi = psi * np.exp(-1j * phase)
            bk = np.broadcast_to(b[k], (N,))
            t = psi.reshape((2,) * N)
            for ax in range(N):   # tensor axis ax = schedule bit ax
                c, s = np.cos(bk[ax]), 1j * np.sin(bk[ax])
                t0, t1 = np.take(t, 0, axis=ax), np.take(t, 1, axis=ax)
                t = np.stack([c * t0 + s * t1, s * t0 + c * t1], axis=ax)
            psi = t.reshape(DIM)
        return psi

    def probabilities(self, params):
        pr = np.abs(self.statevector(params)) ** 2
        return pr / pr.sum()

    def ramp(self, dt):
        """Trotterised linear annealing schedule: gamma_k rises, beta_k falls."""
        k = (np.arange(self.p) + 0.5) / self.p
        return self.join(np.repeat((k * dt)[:, None], self.mg, 1), np.repeat(((1 - k) * dt)[:, None], self.mb, 1))

    def lift_params(self, params, coarse):
        """Embed parameters of a coarser ansatz (same p) into this one."""
        g, b = coarse.split(params)
        G = np.zeros((self.p, self.mg))
        if coarse.cost_mode == 'single':
            G[:] = g[:, :1]
        elif coarse.cost_mode == 'split' and self.cost_mode == 'field':
            G[:, :N], G[:, N] = g[:, :1], g[:, 1]
        else:
            G = g.copy()
        B = np.repeat(b[:, :1], self.mb, 1) if coarse.mb == 1 else b.copy()
        return self.join(G, B)


def make_objective(ansatz, energies, kind='gibbs', alpha=0.25, eta=2.0):
    """Training objectives over the QUBO energies only (never the official cost).

    energy : <E>                                       standard QAOA expectation
    cvar   : mean of the lowest alpha-quantile of E    (Barkoutsos et al. 2020)
    gibbs  : -log <exp(-eta (E - E_min)/std E)>        (Li et al. 2020)
    """
    order = np.argsort(energies)
    e_sorted = energies[order]
    boltz = np.exp(-eta * (energies - e_sorted[0]) / (energies.std() or 1.0))

    def f(params):
        pr = ansatz.probabilities(params)
        if kind == 'energy':
            return float(pr @ energies)
        if kind == 'cvar':
            ps = pr[order]
            take = np.clip(alpha - (np.cumsum(ps) - ps), 0, ps)
            return float(take @ e_sorted / alpha)
        if kind == 'gibbs':
            return float(-np.log(pr @ boltz))
        raise ValueError(kind)
    return f


def train(h, J, energies, p=4, cost_mode='single', mixer_mode='qubit', kind='gibbs', eta=2.0, alpha=0.25,
          max_evals=300, ramp_dts=(0.75, 1.0, 1.25), stage1_evals=40, rhobeg=0.15, init='ramp', init_seed=0):
    """Budgeted training: ramp probes -> COBYLA on standard QAOA -> COBYLA on the richer ansatz.

    Every objective evaluation (including the ramp probes) counts toward max_evals.
    Returns (ansatz, best_params, history).
    """
    final = Ansatz(h, J, p, cost_mode, mixer_mode)
    base = Ansatz(h, J, p, 'single', 'single')
    rich = (cost_mode, mixer_mode) != ('single', 'single')
    history = []
    best = {'x': None, 'v': np.inf, 'ansatz': None}

    def counted(f, a):
        def g(x):
            if len(history) >= max_evals:
                raise BudgetExceeded
            v = f(x)
            history.append(v)
            if v < best['v']:
                best.update(x=np.array(x, float), v=v, ansatz=a)
            return v
        return g

    def cobyla(f, x0, evals, rho):
        if evals <= 0:
            return
        try:
            minimize(f, x0, method='COBYLA', options={'maxiter': evals, 'rhobeg': rho, 'tol': 1e-7})
        except BudgetExceeded:
            pass

    fb = counted(make_objective(base, energies, kind, alpha, eta), base)
    if init == 'ramp':
        starts = [base.ramp(dt) for dt in ramp_dts]
    elif init == 'random':   # ablation: uniform random angles in [0, pi/2)
        rng = np.random.default_rng(init_seed)
        starts = [rng.uniform(0, np.pi / 2, base.n_params) for _ in ramp_dts]
    elif init == 'small':    # ablation: near-identity circuit
        starts = [np.full(base.n_params, 0.05)]
    else:
        raise ValueError(init)
    for x0 in starts:
        fb(x0)
    if not rich:
        cobyla(fb, best['x'], max_evals - len(history), rhobeg)
        return base, best['x'], history
    cobyla(fb, best['x'], stage1_evals - len(history), rhobeg)
    x0 = final.lift_params(best['x'], base)
    ff = counted(make_objective(final, energies, kind, alpha, eta), final)
    ff(x0)  # same state as the stage-1 optimum, now in the richer parameterisation
    cobyla(ff, x0, max_evals - len(history), rhobeg / 2)
    if best['ansatz'] is not final:
        best['x'] = final.lift_params(best['x'], base)
    return final, best['x'], history
