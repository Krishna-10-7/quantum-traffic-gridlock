"""Strict-budget variant: qaoa_nft.train with its one-entry memo DISABLED, so every objective call (including
COBYLA's re-request of its initial point) runs a simulation and is counted against the 300 cap.
Compares with the agent's version on VAL and on random_scenarios(100, seed=2026), via run.solve, seed 0.

    python lab/optimizer_ansatz/_review/review_strict.py
"""
import os
import sys

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import json
import types
import warnings
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
ROOT = AGENT.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import qaoa  # noqa: E402
import run  # noqa: E402
import traffic_model as tm  # noqa: E402

ORIG = qaoa.train


def load_strict():
    src = (AGENT / 'qaoa_nft.py').read_text()
    old = "            if memo['x'] is not None and np.array_equal(x, memo['x']):\n                return memo['v']"
    assert old in src
    src = src.replace(old, "            pass")
    mod = types.ModuleType('qaoa_nft_strict')
    exec(compile(src, 'qaoa_nft_strict', 'exec'), mod.__dict__)
    return mod


def job(s):
    nft = load_strict()
    calls = {'n': 0}
    sv = qaoa.Ansatz.statevector

    def tr(h, J, E, **kw):
        def csv(self, p):
            calls['n'] += 1
            return sv(self, p)
        qaoa.Ansatz.statevector = csv
        try:
            return nft.train(h, J, E, **kw)
        finally:
            qaoa.Ansatz.statevector = sv
    qaoa.train = tr
    try:
        r = run.solve(s, 0, {'cost_mode': 'field', 'stage1_evals': 10})
    finally:
        qaoa.train = ORIG
    q = r['metrics']['baselines']['6_qaoa_raw_samples']
    return {'scenario': s['name'], 'lift': q['lift'], 'enrichment': q['enrichment'], 'p_optimal': q['p_optimal'],
            'hist': len(r['training']['history']), 'sims': calls['n']}


def main():
    out = {}
    R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
    sets = {'VAL': [R] + tm.random_scenarios(60, seed=555), 'g2026': tm.random_scenarios(100, seed=2026)}
    with Pool(2) as pool:
        for k, scs in sets.items():
            rows = pool.map(job, scs)
            L = np.array([r['lift'] for r in rows])
            En = np.array([r['enrichment'] for r in rows])
            out[k] = {'mean_lift': float(L.mean()), 'min_lift': float(L.min()), 'mean_enr': float(En.mean()),
                      'min_enr': float(En.min()), 'R_lift': rows[0]['lift'], 'R_enr': rows[0]['enrichment'],
                      'max_hist': max(r['hist'] for r in rows), 'sims_eq_hist': all(r['sims'] == r['hist'] for r in rows),
                      'rows': rows}
            print(k, json.dumps({a: b for a, b in out[k].items() if a != 'rows'}), flush=True)
    (HERE / 'review_strict.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
