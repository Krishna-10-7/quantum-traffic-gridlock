"""QUBO construction for the two-intersection traffic model.

Modelling rule (scenario independent, see REPORT.md section 2):

  C(x) = sum_k  D_k(x_{S_k})  +  w_sw * sum_n sum_t [x_{n,t} XOR x_{n,t-1}]

where D_k is the exact delay + w_ov * overflow of approach k, which only depends
on the bits S_k the approach can see (its own intersection; plus A's bits for the
linked B approach). The switch term is already quadratic and is kept exactly:
x XOR y = x + y - 2xy.

Each D_k is a pseudo-Boolean function; it has a unique multilinear (Walsh/Fourier)
expansion D_k = sum_S c_S prod_{i in S} z_i (z = 1 - 2x). We keep the terms with |S| <= 2,
i.e. the orthogonal L2(mu) projection of D_k onto quadratic functions. Two measures:

  * 'uniform'  - mu uniform over schedules: the plain second-order Walsh truncation.
  * 'adaptive' - mu proportional to the Boltzmann weight of the surrogate itself,
                 exp(-(E(x) - min E) / (kappa * std E)), solved as a fixed point.
                 This puts the approximation budget where QAOA puts probability mass
                 (low-energy schedules). It uses no knowledge of the true optimum.

The schedule-to-variable mapping is the identity: variable i = schedule bit i
(A1..A4, B1..B4). No auxiliary variables are used.
"""
import itertools

import numpy as np

import traffic_model as tm

N = 8
PAIRS = list(itertools.combinations(range(N), 2))
X = tm.ALL_BITS.astype(float)
FEATURES = np.column_stack([np.ones(256)] + [X[:, i] for i in range(N)] + [X[:, i] * X[:, j] for i, j in PAIRS])


def approach_tables(s):
    """Exact per-approach cost tables over all 256 joint bit patterns.

    Approaches that only see 4 bits are constant along the other 4, so their
    projection only couples their own bits (sparsity follows the traffic topology).
    """
    tables = {}
    for idx, row in enumerate(tm.ALL_BITS):
        for k, v in tm.approach_costs(s, list(row)).items():
            tables.setdefault(k, np.zeros(256))[idx] = v
    return tables


def _project(values, weights):
    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(FEATURES * sw[:, None], values * sw, rcond=None)
    return coef


def coef_to_qubo(coef):
    """Feature coefficients -> (offset, upper-triangular Q) with E(x) = offset + x^T Q x."""
    Q = np.zeros((N, N))
    for i in range(N):
        Q[i, i] = coef[1 + i]
    for m, (i, j) in enumerate(PAIRS):
        Q[i, j] = coef[1 + N + m]
    return float(coef[0]), Q


def qubo_energy(Q, offset, bits):
    x = np.asarray(bits, dtype=float)
    return float(offset + x @ Q @ x)


def energies(Q, offset):
    """Energy of every schedule, index = int(bitstring, 2)."""
    return offset + np.einsum('si,ij,sj->s', X, Q, X)


def build_qubo(s, measure='webster', kappa=1.0, iters=8):
    """measure: 'webster' (default, closed-form local rule in analytic_qubo.py, no 256-schedule table),
    'adaptive' / 'uniform' (weighted-L2 projection of the full cost table; ablations / reference)."""
    if measure in ('webster', 'red', 'local_gibbs'):
        import analytic_qubo
        return analytic_qubo.build_analytic_qubo(s, rule=measure)
    tables = approach_tables(s)
    total = sum(tables.values())
    w = np.ones(256)
    coef = _project(total, w)
    if measure == 'adaptive':
        for _ in range(iters):
            e = FEATURES @ coef
            if e.std() < 1e-12:  # flat landscape: nothing to reweight
                break
            w = np.exp(-(e - e.min()) / (kappa * e.std()))
            coef = _project(total, w)
    elif measure != 'uniform':
        raise ValueError(measure)
    offset, Q = coef_to_qubo(coef)
    # numerically exact zeros for couplings the topology forbids (e.g. A-B pairs that the link does not create)
    Q[np.abs(Q) < 1e-9] = 0.0
    return offset, Q


def qubo_to_ising(Q, offset):
    """x_i = (1 - z_i)/2. Returns (h, J, const) with E = const + sum h_i z_i + sum_{i<j} J_ij z_i z_j."""
    h = np.zeros(N)
    J = np.zeros((N, N))
    const = offset
    for i in range(N):
        a = Q[i, i]
        const += a / 2
        h[i] -= a / 2
    for i, j in PAIRS:
        b = Q[i, j]
        if b == 0:
            continue
        const += b / 4
        h[i] -= b / 4
        h[j] -= b / 4
        J[i, j] += b / 4
    return h, J, const
