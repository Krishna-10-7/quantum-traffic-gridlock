"""Post-hoc sensitivity of the analytic rule to its two shape constants (clip, platoon-aware demand).

    python lab/analytic_qubo/clip_sensitivity.py

Surrogate-level only (QUBO exact minimizer regret, Pearson). The frozen rule (clip 0.25, link-aware demand)
was chosen on random_scenarios(200, seed=31); this table is reported on the tuning set AND on VAL
for transparency, it is NOT used to re-select anything.
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

GRID = [(0.15, True), (0.2, True), (0.25, True), (0.3, True), (0.35, True), (0.4, True), (0.25, False), (0.5, True)]


def table(scs, costs):
    out = {}
    for clip, link in GRID:
        reg, pr = [], []
        for s, c in zip(scs, costs):
            off, Q = aq.build_analytic_qubo(s, clip=clip, link=link)
            E = qb.energies(Q, off)
            reg.append((c[int(np.argmin(E))] - c.min()) / c.min())
            pr.append(np.corrcoef(E, c)[0, 1])
        reg = np.array(reg)
        out[f'clip={clip} link={link}'] = {'frac_le_5pct': float((reg <= 0.05).mean()), 'mean_regret': float(reg.mean()),
                                           'max_regret': float(reg.max()), 'frac_zero': float((reg == 0).mean()),
                                           'mean_pearson': float(np.mean(pr))}
    return out


def main():
    sets = {'tune(seed31,n200)': tm.random_scenarios(200, seed=31),
            'VAL': [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(60, seed=555)}
    res = {}
    for name, scs in sets.items():
        costs = [tm.cost_table(s) for s in scs]
        res[name] = table(scs, costs)
        ref = {}
        for m in ('adaptive', 'uniform'):
            reg = []
            for s, c in zip(scs, costs):
                off, Q = qb.build_qubo(s, measure=m)
                reg.append((c[int(np.argmin(qb.energies(Q, off)))] - c.min()) / c.min())
            reg = np.array(reg)
            ref[f'repo {m}'] = {'frac_le_5pct': float((reg <= 0.05).mean()), 'mean_regret': float(reg.mean()),
                                'max_regret': float(reg.max()), 'frac_zero': float((reg == 0).mean())}
        res[name].update(ref)
        print(f'\n{name}')
        for k, v in res[name].items():
            print(f"  {k:<24} <=5% {v['frac_le_5pct']:.3f}  =0 {v['frac_zero']:.3f}  mean {v['mean_regret']*100:5.2f}%  "
                  f"max {v['max_regret']*100:5.1f}%" + (f"  Pearson {v['mean_pearson']:.4f}" if 'mean_pearson' in v else ''))
    (HERE / 'results' / 'clip_sensitivity.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
