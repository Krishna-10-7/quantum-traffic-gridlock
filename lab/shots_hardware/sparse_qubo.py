"""Hardware-sparse QUBOs (same modelling rule, restricted coupling support).

The repo QUBO is E = argmin_{g in V} sum_x mu(x) (C(x) - g(x))^2 with V = all quadratic pseudo-Boolean
functions on 8 bits (28 couplings) and the adaptive measure mu = mu_kappa(E) (fixed point). Here V is
replaced by V_S = span{1, x_i, x_i x_j : (i, j) in S} for a coupling support S, i.e. the SAME projection
onto a smaller function space ("prune-and-refit", the analogue of a relaxed lasso refit):

  * 'prune'      : S_tau = {(i,j) : |Q_ij| >= tau max|Q|} from the full QUBO, couplings simply deleted
                   (no refit)  - this is what experiments/noise.py does;
  * 'refit'      : same S_tau, but all remaining coefficients re-projected onto V_{S_tau}
                   (the weighted-L2 optimum within the sparse space);
  * 'tree'       : fixed, scenario-independent support with ZERO SWAPs on heavy-hex:
                   S_tree = {A1A2, A2A3, A3A4, B1B2, B2B3, B3B4, A1B2}
                   = the six adjacent-slot pairs that carry the exact switching term, plus the link edge
                   A1 -> B2 (slot-1 departures of A arrive at B in slot 2). It is a tree with one
                   degree-3 node (B2), hence a subgraph of the heavy-hex lattice (girth 12, degree <= 3).
"""
import numpy as np

import common as C

qb = C.qb
PAIRS = qb.PAIRS
TREE = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (0, 5)]


def project_support(s, support, kappa=1.0, iters=8):
    """Adaptive weighted-L2 projection of the exact cost onto V_S. support: iterable of (i, j), i < j."""
    tables = qb.approach_tables(s)
    total = sum(tables.values())
    sup = set(map(tuple, support))
    cols = list(range(1 + qb.N)) + [1 + qb.N + m for m, pr in enumerate(PAIRS) if pr in sup]
    F = qb.FEATURES[:, cols]

    def proj(w):
        sw = np.sqrt(w)
        c, *_ = np.linalg.lstsq(F * sw[:, None], total * sw, rcond=None)
        return c
    w = np.ones(256)
    c = proj(w)
    for _ in range(iters):
        e = F @ c
        if e.std() < 1e-12:
            break
        w = np.exp(-(e - e.min()) / (kappa * e.std()))
        c = proj(w)
    full = np.zeros(qb.FEATURES.shape[1])
    full[cols] = c
    off, Q = qb.coef_to_qubo(full)
    Q[np.abs(Q) < 1e-9] = 0.0
    return off, Q


def sparse_qubo(s, mode='full', tau=0.0):
    cfg = C.run.DEFAULTS
    off, Q = qb.build_qubo(s, measure=cfg['measure'], kappa=cfg['kappa'])
    if mode == 'full' or (mode in ('prune', 'refit') and tau <= 0):
        return off, Q
    if mode == 'tree':
        return project_support(s, TREE, kappa=cfg['kappa'])
    od = np.triu(np.abs(Q), 1)
    keep = [(i, j) for i, j in PAIRS if od[i, j] >= tau * od.max() and od[i, j] > 0]
    if mode == 'prune':
        Qp = np.where(np.triu(np.ones_like(Q), 1).astype(bool), 0.0, Q)
        for i, j in keep:
            Qp[i, j] = Q[i, j]
        return off, Qp
    if mode == 'refit':
        return project_support(s, keep, kappa=cfg['kappa'])
    raise ValueError(mode)


def regret(s, off, Q, costs=None):
    costs = C.tm.cost_table(s) if costs is None else costs
    e = qb.energies(Q, off)
    k = int(np.argmin(e))
    return float((costs[k] - costs.min()) / costs.min()) if costs.min() > 0 else 0.0


if __name__ == '__main__':
    import sys
    sc = C.tune_set(20) if 'val' not in sys.argv else C.val_set()
    for mode, tau in [('full', 0), ('prune', 0.1), ('refit', 0.1), ('prune', 0.25), ('refit', 0.25), ('tree', 0)]:
        rg, nc, pe = [], [], []
        for s in sc:
            costs = C.tm.cost_table(s)
            off, Q = sparse_qubo(s, mode, tau)
            rg.append(regret(s, off, Q, costs))
            nc.append(int((np.triu(Q, 1) != 0).sum()))
            pe.append(np.corrcoef(qb.energies(Q, off), costs)[0, 1])
        rg = np.array(rg)
        print(f'{mode:6s} tau={tau:<5} couplings {np.mean(nc):5.1f}  regret mean {rg.mean():.4f} max {rg.max():.3f} '
              f'frac<=5% {np.mean(rg <= 0.05 + 1e-12):.2f}  pearson {np.mean(pe):.3f}')
