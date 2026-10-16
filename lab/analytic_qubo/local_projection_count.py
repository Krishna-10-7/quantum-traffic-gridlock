"""How many single-approach simulations does the CURRENT projection QUBO need?

    python lab/analytic_qubo/local_projection_count.py

The repo's qubo.build_qubo builds per-approach tables by calling traffic_model.approach_costs on all 256
joint schedules (256 x 4 = 1,024 single-approach recursions). Projection is linear, and every approach
cost D_k depends only on its own window S_k, so the same QUBO can be assembled from LOCAL tables:

    naive local windows   : A-NS 16 + A-EW 16 + unlinked B 16 + linked B 2^8 = 256        -> 304 recursions
    exact linked window   : the linked B approach does not see A4 (its platoon leaves the model),
                            so 2^7 = 128 patterns suffice                                 -> 176 recursions

This script rebuilds BOTH repo projections (uniform and adaptive, kappa = 1, 8 fixed-point iterations)
from the 176 local recursions only and checks that the energies match qubo.build_qubo to round-off on
VAL = R + random_scenarios(60, seed=555). It also reports what each rule still has to sum over:

  * uniform projection  : decomposes per approach (sums of <= 2^7 terms), like the analytic rule;
  * adaptive projection : its Gibbs weights mu(x) ~ exp(-E(x)/(kappa sigma_E)) do not factorise, so the normal
                          equations sum the assembled cost sum_k D_k(x_{S_k}) over all 2^8 joint schedules
                          (the 256-entry cost table, assembled from local pieces, with no extra simulator calls);
  * analytic Webster rule: product measure, per-approach sums only; never assembles C(x).
"""
import os
import sys
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import qubo as qb  # noqa: E402
import traffic_model as tm  # noqa: E402
import analytic_qubo as aq  # noqa: E402

X = tm.ALL_BITS


def local_joint_values(s):
    """Per-approach values on the 256 joint schedules, looked up from the 176 local recursions."""
    aq.COUNTER['approach_sims'] = 0
    comps = aq.local_components(s)
    sims = aq.COUNTER['approach_sims']
    total = np.zeros(256)
    for name, vars_, pats, vals in comps:
        k = len(vars_)
        idx = X[:, vars_] @ (2 ** np.arange(k - 1, -1, -1))     # local pattern index (patterns are MSB-first)
        total += vals[idx]
    sw = s['switch_weight'] * sum((X[:, n + t] != X[:, n + t - 1]) for n in (0, 4) for t in range(1, 4))
    return total + sw, sims


def local_projection(s, measure='adaptive', kappa=1.0, iters=8):
    total, sims = local_joint_values(s)
    w = np.ones(256)
    coef = qb._project(total, w)
    if measure == 'adaptive':
        for _ in range(iters):
            e = qb.FEATURES @ coef
            if e.std() < 1e-12:
                break
            w = np.exp(-(e - e.min()) / (kappa * e.std()))
            coef = qb._project(total, w)
    off, Q = qb.coef_to_qubo(coef)
    Q[np.abs(Q) < 1e-9] = 0.0
    return off, Q, sims


def main():
    scs = [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(60, seed=555)
    worst = {'uniform': 0.0, 'adaptive': 0.0}
    sims_seen = set()
    for s in scs:
        assert np.allclose(local_joint_values(s)[0], tm.cost_table(s)), 'local decomposition != scorer'
        for m in worst:
            o, Q, sims = local_projection(s, m)
            o2, Q2 = qb.build_qubo(s, measure=m, kappa=1.0)
            worst[m] = max(worst[m], float(np.abs(qb.energies(Q, o) - qb.energies(Q2, o2)).max()))
            sims_seen.add(sims)
    print(f'VAL scenarios: {len(scs)}; single-approach recursions per scenario (local windows): {sorted(sims_seen)}')
    print('  naive windows (linked B over all 8 bits): 16 + 16 + 16 + 256 = 304; exact windows: 16 + 16 + 16 + 128 = 176')
    print('  repo implementation (approach_costs on 256 joint schedules): 256 x 4 = 1024')
    for m, v in worst.items():
        print(f'  {m:<8} projection rebuilt from local tables: max |E_local - E_repo| = {v:.2e}')
    print('local decomposition sum_k D_k + switching == scorer cost table on all VAL scenarios: OK')


if __name__ == '__main__':
    main()
