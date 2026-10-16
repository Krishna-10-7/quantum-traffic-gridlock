"""Diagnostics for 'what changes' from 8 to 12 qubits (tuning set, chain generator seed 31).

  python lab/chain_extension/diagnose.py budget      # exact-distribution lift vs evaluation budget (600/1000 are
                                                     # DIAGNOSTIC ONLY - above the 300 cap, never used in the pipeline)
  python lab/chain_extension/diagnose.py subproblem  # same traffic: official 8-qubit A,B problem (repo run.solve)
                                                     # vs the 12-qubit chain (chain_run.solve_chain)
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

import numpy as np  # noqa: E402

import chain_baselines as cb  # noqa: E402
import chain_model as cm  # noqa: E402
import chain_qaoa as cq  # noqa: E402
import chain_qubo as cqb  # noqa: E402
import chain_run as cr  # noqa: E402
import chain_scorer as cs  # noqa: E402

RES = HERE / 'results'
BUDGETS = (100, 200, 300, 600, 1000)


def _budget_job(s):
    costs = cm.cost_table(s)
    off, Q, _ = cqb.build_qubo(s)
    h, J, _ = cqb.qubo_to_ising(Q, off)
    e = cqb.energies(Q, off)
    out = {}
    for mode in ('final', 'standard'):
        kw = dict(cost_mode='split', mixer_mode='qubit') if mode == 'final' else dict(cost_mode='single', mixer_mode='single')
        for B in BUDGETS:
            a, x, hist = cq.train(h, J, e, p=4, max_evals=B, **kw)
            assert len(hist) <= B
            out[f'{mode}@{B}'] = cb.distribution_metrics(costs, a.probabilities(x))['lift']
    return out


def _sub_job(s):
    import run  # repo pipeline, unchanged
    r8 = run.solve(cs.restrict_AB(s), 0)['metrics']['baselines']['6_qaoa_raw_samples']
    r12 = cr.summarize(cr.solve_chain(s, 0))
    return {'scenario': s['name'], 'lift8': r8['lift'], 'enr8': r8['enrichment'], 'popt8': r8['p_optimal'],
            'lift12': r12['lift'], 'enr12': r12['enrichment'], 'popt12': r12['p_optimal']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['budget', 'subproblem'])
    ap.add_argument('--n', type=int, default=12)
    a = ap.parse_args()
    scen = cm.random_chain_scenarios(a.n, seed=31)
    RES.mkdir(exist_ok=True)
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(_budget_job if a.cmd == 'budget' else _sub_job, scen))
    if a.cmd == 'budget':
        summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(json.dumps(summ, indent=1))
        (RES / 'diag_budget_seed31.json').write_text(json.dumps({'mean_exact_lift': summ, 'rows': rows}, indent=1))
    else:
        summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'scenario'}
        summ.update({'min_lift8': float(min(r['lift8'] for r in rows)), 'min_lift12': float(min(r['lift12'] for r in rows))})
        print(json.dumps(summ, indent=1))
        (RES / 'diag_subproblem_seed31.json').write_text(json.dumps({'mean': summ, 'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
