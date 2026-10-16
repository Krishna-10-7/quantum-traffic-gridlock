"""Full-pipeline evaluation of QUBO rules: run.solve with the QUBO builder swapped, everything else frozen.

    python lab/analytic_qubo/qaoa_eval.py --set tune --rules adaptive webster local_gibbs uniform
    python lab/analytic_qubo/qaoa_eval.py --set val  --rules adaptive webster

`adaptive` is the repo's current final config (run.solve(s, 0) unchanged). Other rules replace
qubo.build_qubo by analytic_qubo.build_analytic_qubo(s, rule) inside run.solve; QAOA config
(p=4, split cost, per-qubit mixer, Gibbs eta=2, 40+260 COBYLA evals, 4096 shots, seed 0) unchanged.
Metric: 4096-shot sampled metrics with sampling seed 0 (as run.solve does).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import qubo as qb  # noqa: E402
import run  # noqa: E402
import traffic_model as tm  # noqa: E402
import analytic_qubo as aq  # noqa: E402

_ORIG = qb.build_qubo


def _patched(rule, **kw):
    def f(s, measure='adaptive', kappa=1.0, **_):
        return aq.build_analytic_qubo(s, rule=rule, **kw)
    return f


RULES = {
    'adaptive': None,                                   # current repo final config
    'uniform_repo': ('repo_uniform', {}),
    'webster': ('webster', {}),                         # final analytic rule: clip 0.25, link-aware demand
    'webster_v0': ('webster', {'clip': 0.3, 'link': False}),   # first version (tuning run only)
    'uniform': ('uniform', {}),
    'local_gibbs': ('local_gibbs', {}),
    'red': ('red', {}),
}


def job(args):
    s, seed, rule = args
    spec = RULES[rule]
    if spec is None:
        qb.build_qubo = _ORIG
        cfg = None
    elif spec[0] == 'repo_uniform':
        qb.build_qubo = _ORIG
        cfg = {'measure': 'uniform'}
    else:
        qb.build_qubo = _patched(spec[0], **spec[1])
        cfg = None
    res = run.solve(s, seed, cfg)
    qb.build_qubo = _ORIG
    b = res['metrics']['baselines']
    q = b['6_qaoa_raw_samples']
    return {'scenario': s.get('name'), 'rule': rule, 'seed': seed, 'C_star': q['C_star'], 'C_rand': q['C_rand'],
            'E_Q': q['E_Q'], 'lift': q['lift'], 'enrichment': q['enrichment'], 'p_optimal': q['p_optimal'],
            'regret_rel': res['metrics']['surrogate']['regret_rel'],
            'pearson': res['metrics']['surrogate']['pearson_qubo_vs_official'],
            'spearman': res['metrics']['surrogate']['spearman_qubo_vs_official'],
            'exact_lift': res['metrics']['qaoa_exact_distribution']['lift'],
            'exact_enrichment': res['metrics']['qaoa_exact_distribution']['enrichment'],
            'qaoa_schedule_cost': res['schedule']['official_cost'],
            'evals': res['metrics']['training']['objective_evaluations'],
            'seconds': res['metrics']['runtime_seconds']['total']}


def scenario_set(name):
    if name == 'tune':
        return tm.random_scenarios(60, seed=31)
    if name == 'val':
        return [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(60, seed=555)
    raise ValueError(name)


def aggregate(rows):
    reg = np.array([r['regret_rel'] for r in rows])
    lift = np.array([r['lift'] for r in rows])
    R = [r for r in rows if r['scenario'] == 'R']
    out = {'n': len(rows), 'mean_lift': float(lift.mean()), 'min_lift': float(lift.min()),
           'mean_enrichment': float(np.mean([r['enrichment'] for r in rows])),
           'mean_p_optimal': float(np.mean([r['p_optimal'] for r in rows])),
           'mean_exact_lift': float(np.mean([r['exact_lift'] for r in rows])),
           'mean_regret': float(reg.mean()), 'max_regret': float(reg.max()),
           'frac_regret_le_5pct': float((reg <= 0.05).mean()), 'frac_regret_zero': float((reg == 0).mean()),
           'regret_points_of_10': float(np.mean(np.clip((0.25 - reg) / 0.20, 0, 1)) * 10),
           'mean_pearson': float(np.mean([r['pearson'] for r in rows])),
           'mean_spearman': float(np.mean([r['spearman'] for r in rows])),
           'frac_schedule_optimal': float(np.mean([r['qaoa_schedule_cost'] == r['C_star'] for r in rows])),
           'max_evals': int(max(r['evals'] for r in rows)),
           'mean_seconds': float(np.mean([r['seconds'] for r in rows]))}
    if R:
        out['R_lift'] = R[0]['lift']
        out['R_enrichment'] = R[0]['enrichment']
        out['R_p_optimal'] = R[0]['p_optimal']
        out['R_regret'] = R[0]['regret_rel']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='tune', choices=['tune', 'val'])
    ap.add_argument('--rules', nargs='+', default=['adaptive', 'webster'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    scs = scenario_set(a.set)
    out_dir = HERE / 'results'
    out_dir.mkdir(exist_ok=True)
    jobs = [(s, a.seed, r) for r in a.rules for s in scs]
    t = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, jobs, chunksize=2))
    summary = {}
    for r in a.rules:
        rr = [x for x in rows if x['rule'] == r]
        summary[r] = aggregate(rr)
        (out_dir / f'qaoa_{a.set}_{r}_seed{a.seed}.json').write_text(json.dumps({'summary': summary[r], 'rows': rr}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f'wall {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
