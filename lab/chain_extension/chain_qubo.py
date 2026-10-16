"""QUBO for the A->B->C chain by the SAME modelling rule as qubo.py (weighted-L2 quadratic projection).

    E = argmin_{g quadratic} sum_x mu(x) (C(x) - g(x))^2,   g(x) = offset + sum_{i<=j} Q_ij x_i x_j,  x in {0,1}^12

  * measure='uniform'  : mu uniform  ==  second-order Walsh truncation. Because the uniform measure is a product
                         measure, the projection of each approach term D_k only involves the bits of its light-cone
                         scope S_k (chain_model.scopes), so it can be computed LOCALLY from 2^|S_k| simulations
                         (build_qubo_local_uniform); the result is identical to the global fit (tested).
  * measure='adaptive' : mu_kappa(x) ~ exp(-(E(x) - min E)/(kappa std E)) solved as a fixed point (8 iterations),
                         exactly as in qubo.py. The weights depend only on the surrogate's own energies.
  * sparsity='lightcone' (optional): restrict the quadratic features to pairs inside some light-cone scope
                         (chain_model.lightcone_pairs); the adaptive fit then has the topology's coupling graph.

Variable i = schedule bit i (A1..A4, B1..B4, C1..C4); no auxiliary variables.
"""
import itertools

import numpy as np

import chain_model as cm


def bit_matrix(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)


class Basis:
    """Quadratic feature map on n bits restricted to a list of pairs."""

    def __init__(self, n, pairs=None):
        self.n = n
        self.pairs = list(itertools.combinations(range(n), 2)) if pairs is None else list(pairs)
        self.X = bit_matrix(n)
        self.F = np.column_stack([np.ones(len(self.X))] + [self.X[:, i] for i in range(n)]
                                 + [self.X[:, i] * self.X[:, j] for i, j in self.pairs])

    def to_qubo(self, coef):
        Q = np.zeros((self.n, self.n))
        Q[np.arange(self.n), np.arange(self.n)] = coef[1:1 + self.n]
        for m, (i, j) in enumerate(self.pairs):
            Q[i, j] = coef[1 + self.n + m]
        return float(coef[0]), Q


def _project(F, values, weights):
    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(F * sw[:, None], values * sw, rcond=None)
    return coef


def energies(Q, offset, X=None):
    X = bit_matrix(Q.shape[0]) if X is None else X
    return offset + np.einsum('si,ij,sj->s', X, Q, X)


def build_qubo(s, measure='adaptive', kappa=1.0, iters=8, sparsity='dense', total=None):
    """Global construction (4096 configurations). Returns (offset, Q, info)."""
    pairs = cm.lightcone_pairs(s) if sparsity == 'lightcone' else None
    basis = Basis(cm.N, pairs)
    if total is None:
        total = cm.cost_table(s)
    w = np.ones(len(total))
    coef = _project(basis.F, total, w)
    ess = float(len(total))
    if measure == 'adaptive':
        for _ in range(iters):
            e = basis.F @ coef
            if e.std() < 1e-12:
                break
            w = np.exp(-(e - e.min()) / (kappa * e.std()))
            coef = _project(basis.F, total, w)
        ess = float(w.sum() ** 2 / (w ** 2).sum())
    elif measure != 'uniform':
        raise ValueError(measure)
    offset, Q = basis.to_qubo(coef)
    Q[np.abs(Q) < 1e-9] = 0.0
    return offset, Q, {'simulations': len(total), 'effective_sample_size': ess,
                       'nonzero_couplings': int((np.abs(np.triu(Q, 1)) > 0).sum())}


def build_qubo_local_uniform(s):
    """Approach-wise construction of the uniform-measure QUBO using only 2^|S_k| local simulations per approach.

    For approach k with scope S_k (m bits), D_k is evaluated on the 2^m local configurations (other bits fixed to 0,
    which is legitimate because D_k does not depend on them), projected onto quadratic functions of those m bits under
    the uniform measure, and the coefficients are scattered into the global Q. The switching term is quadratic and
    added exactly. Returns (offset, Q, info)."""
    n = cm.N
    Q = np.zeros((n, n))
    offset = 0.0
    sims = 0
    for k, sc in cm.scopes(s).items():
        m = len(sc)
        local = Basis(m)
        full = np.zeros((2 ** m, n), dtype=np.int8)
        full[:, sc] = local.X.astype(np.int8)
        vals = cm.approach_tables(s, full)[k]
        sims += 2 ** m
        off_k, Q_k = local.to_qubo(_project(local.F, vals, np.ones(2 ** m)))
        offset += off_k
        for a in range(m):
            for b in range(a, m):
                Q[sc[a], sc[b]] += Q_k[a, b]
    w = s['switch_weight']
    for node in cm.NODES:
        o = cm.NODE_OFF[node]
        for t in range(1, 4):           # XOR(x, y) = x + y - 2xy
            Q[o + t, o + t] += w
            Q[o + t - 1, o + t - 1] += w
            Q[o + t - 1, o + t] -= 2 * w
    Q[np.abs(Q) < 1e-9] = 0.0
    return offset, Q, {'simulations': sims, 'nonzero_couplings': int((np.abs(np.triu(Q, 1)) > 0).sum())}


def qubo_to_ising(Q, offset):
    """x_i = (1 - z_i)/2 -> (h, J upper, const) with E = const + sum h_i z_i + sum_{i<j} J_ij z_i z_j."""
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    const = offset
    for i in range(n):
        a = Q[i, i]
        const += a / 2
        h[i] -= a / 2
    for i, j in itertools.combinations(range(n), 2):
        b = Q[i, j]
        if b == 0:
            continue
        const += b / 4
        h[i] -= b / 4
        h[j] -= b / 4
        J[i, j] += b / 4
    return h, J, const
