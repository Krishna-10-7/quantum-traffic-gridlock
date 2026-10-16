"""n-qubit version of qaoa.py (same ansatz, same objectives, same budgeted two-stage trainer).

The only change from qaoa.py is that the number of qubits n is read from the Hamiltonian instead of being the
module constant 8. With n = 8 this reproduces qaoa.Ansatz / qaoa.train bit for bit (tests/test_chain.py).
Objectives are imported unchanged from qaoa.make_objective (QUBO energies only, never the official cost).
"""
import itertools

import numpy as np
from scipy.optimize import minimize

import qaoa as q8   # repo module: make_objective, BudgetExceeded

COST_WIDTH = {'single': lambda n: 1, 'split': lambda n: 2, 'field': lambda n: n + 1}
MIXER_WIDTH = {'single': lambda n: 1, 'qubit': lambda n: n}
_ZCACHE = {}


def zmatrix(n):
    if n not in _ZCACHE:
        _ZCACHE[n] = 1 - 2 * np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    return _ZCACHE[n]


class Ansatz:
    def __init__(self, h, J, p=4, cost_mode='single', mixer_mode='single'):
        self.h, self.J, self.p = np.asarray(h, float), np.asarray(J, float), p
        self.n = len(self.h)
        self.dim = 2 ** self.n
        self.cost_mode, self.mixer_mode = cost_mode, mixer_mode
        Z = zmatrix(self.n)
        field = Z * self.h
        coup = np.einsum('si,ij,sj->s', Z, self.J, Z)
        self.diag = field.sum(1) + coup
        self.scale = float(self.diag.std()) or 1.0
        if cost_mode == 'single':
            T = self.diag[:, None]
        elif cost_mode == 'split':
            T = np.column_stack([field.sum(1), coup])
        else:
            T = np.column_stack([field, coup])
        self.T = T / self.scale
        self.mg, self.mb = COST_WIDTH[cost_mode](self.n), MIXER_WIDTH[mixer_mode](self.n)

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
        g, b = self.split(params)
        n = self.n
        out = []
        for k in range(self.p):
            if self.cost_mode == 'single':
                gf, gj = np.full(n, g[k, 0]), g[k, 0]
            elif self.cost_mode == 'split':
                gf, gj = np.full(n, g[k, 0]), g[k, 1]
            else:
                gf, gj = g[k, :n], g[k, n]
            out.append((gf / self.scale, gj / self.scale, np.broadcast_to(b[k], (n,)).copy()))
        return out

    def statevector(self, params):
        g, b = self.split(params)
        n = self.n
        psi = np.full(self.dim, 1 / np.sqrt(self.dim), dtype=complex)
        for k in range(self.p):
            phase = self.T @ g[k] if self.mg > 1 else self.T[:, 0] * g[k, 0]
            psi = psi * np.exp(-1j * phase)
            bk = np.broadcast_to(b[k], (n,))
            t = psi.reshape((2,) * n)
            for ax in range(n):
                c, s = np.cos(bk[ax]), 1j * np.sin(bk[ax])
                t0, t1 = np.take(t, 0, axis=ax), np.take(t, 1, axis=ax)
                t = np.stack([c * t0 + s * t1, s * t0 + c * t1], axis=ax)
            psi = t.reshape(self.dim)
        return psi

    def probabilities(self, params):
        pr = np.abs(self.statevector(params)) ** 2
        return pr / pr.sum()

    def ramp(self, dt):
        k = (np.arange(self.p) + 0.5) / self.p
        return self.join(np.repeat((k * dt)[:, None], self.mg, 1), np.repeat(((1 - k) * dt)[:, None], self.mb, 1))

    def lift_params(self, params, coarse):
        g, b = coarse.split(params)
        n = self.n
        G = np.zeros((self.p, self.mg))
        if coarse.cost_mode == 'single':
            G[:] = g[:, :1]
        elif coarse.cost_mode == 'split' and self.cost_mode == 'field':
            G[:, :n], G[:, n] = g[:, :1], g[:, 1]
        else:
            G = g.copy()
        B = np.repeat(b[:, :1], self.mb, 1) if coarse.mb == 1 else b.copy()
        return self.join(G, B)


make_objective = q8.make_objective
BudgetExceeded = q8.BudgetExceeded


def train(h, J, energies, p=4, cost_mode='single', mixer_mode='qubit', kind='gibbs', eta=2.0, alpha=0.25,
          max_evals=300, ramp_dts=(0.75, 1.0, 1.25), stage1_evals=40, rhobeg=0.15, init='ramp', init_seed=0):
    """Identical control flow to qaoa.train; every objective evaluation (probes included) counts toward max_evals."""
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
    elif init == 'random':
        rng = np.random.default_rng(init_seed)
        starts = [rng.uniform(0, np.pi / 2, base.n_params) for _ in ramp_dts]
    elif init == 'small':
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
    ff(x0)
    cobyla(ff, x0, max_evals - len(history), rhobeg / 2)
    if best['ansatz'] is not final:
        best['x'] = final.lift_params(best['x'], base)
    return final, best['x'], history
