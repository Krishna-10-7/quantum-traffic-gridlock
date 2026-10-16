"""Coefficient-level agreement between the analytic (Webster-point) QUBO and the repo's adaptive projection.

    python lab/analytic_qubo/coef_compare.py

For each VAL scenario both QUBOs are mapped to Ising form (h, J) and compared:
  * Pearson correlation of the 8 + 28 Ising coefficients;
  * relative L2 distance ||theta_an - theta_ad|| / ||theta_ad||;
  * the largest |J| that the analytic rule sets to exactly 0 (A4-B pairs) has in the adaptive QUBO.
Prints scenario R's analytic QUBO with the traffic meaning of each coupling class.
"""
import json
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

IU = np.triu_indices(8, 1)
NAMES = [f'{n}{t}' for n in 'AB' for t in range(1, 5)]


def ising_vec(off, Q):
    h, J, _ = qb.qubo_to_ising(Q, off)
    return np.r_[h, J[IU]]


def main():
    scs = [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(60, seed=555)
    rows = []
    for s in scs:
        va = ising_vec(*aq.build_analytic_qubo(s))
        vd = ising_vec(*qb.build_qubo(s, measure='adaptive', kappa=1.0))
        vu = ising_vec(*qb.build_qubo(s, measure='uniform'))
        Jd = vd[8:]
        a4b = [m for m, (i, j) in enumerate(zip(*IU)) if i == 3 and j >= 4]
        rows.append({'scenario': s.get('name'),
                     'pearson_vs_adaptive': float(np.corrcoef(va, vd)[0, 1]),
                     'rel_l2_vs_adaptive': float(np.linalg.norm(va - vd) / np.linalg.norm(vd)),
                     'pearson_uniform_vs_adaptive': float(np.corrcoef(vu, vd)[0, 1]),
                     'rel_l2_uniform_vs_adaptive': float(np.linalg.norm(vu - vd) / np.linalg.norm(vd)),
                     'adaptive_A4B_max_over_maxJ': float(np.abs(Jd[a4b]).max() / np.abs(Jd).max())})
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'scenario'}
    agg['min_pearson_vs_adaptive'] = float(np.min([r['pearson_vs_adaptive'] for r in rows]))
    print(json.dumps(agg, indent=1))
    (HERE / 'results' / 'coef_compare_val.json').write_text(json.dumps({'summary': agg, 'rows': rows}, indent=1))

    R = scs[0]
    off, Q = aq.build_analytic_qubo(R)
    np.set_printoptions(precision=2, suppress=True, linewidth=140)
    print('\nScenario R, analytic QUBO (E = offset + x^T Q x), offset = %.2f' % off)
    print('      ' + '  '.join(f'{n:>6}' for n in NAMES))
    for i in range(8):
        print(f'{NAMES[i]:>5} ' + '  '.join(f'{Q[i, j]:6.2f}' if j >= i else '      ' for j in range(8)))
    print('A-A / B-B: same-intersection interference + exact switching (-2 w_sw on consecutive slots);'
          ' A1..A3 x B: spillback / green-wave; A4 x B = 0 exactly (A4 platoon leaves the model).')


if __name__ == '__main__':
    main()
