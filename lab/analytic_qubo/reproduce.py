"""Reproduce the headline numbers of the analytic (Webster-point Taylor) QUBO track.

    python lab/analytic_qubo/reproduce.py            # closed-form check + surrogate metrics on VAL (~1 min)
    python lab/analytic_qubo/reproduce.py --qaoa     # + full QAOA pipeline on VAL, current vs analytic (~6 min, 2 workers)
    python lab/analytic_qubo/reproduce.py --seeds    # + scenario R, sampling seeds 0-4, current vs analytic

VAL = [scenario R] + traffic_model.random_scenarios(60, seed=555).  QAOA metric: 4096-shot sampled
metrics with sampling seed 0, exactly as run.solve computes them.  The exact cost table is used
ONLY to score (regret, Pearson, lift); the analytic QUBO never sees it.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import qubo as qb  # noqa: E402
import traffic_model as tm  # noqa: E402
import analytic_qubo as aq  # noqa: E402
import closed_form  # noqa: E402
import qaoa_eval  # noqa: E402

RULES = ['adaptive', 'webster', 'uniform', 'local_gibbs', 'red']


def build(s, rule):
    if rule == 'adaptive':
        return qb.build_qubo(s, measure='adaptive', kappa=1.0)
    return aq.build_analytic_qubo(s, rule=rule)


def surrogate_row(s, rule):
    costs = tm.cost_table(s)                     # scoring only
    off, Q = build(s, rule)
    E = qb.energies(Q, off)
    k = int(np.argmin(E))
    cstar = costs.min()
    rk = lambda v: np.argsort(np.argsort(v))
    return {'scenario': s.get('name'), 'regret': float((costs[k] - cstar) / cstar) if cstar > 0 else 0.0,
            'pearson': float(np.corrcoef(E, costs)[0, 1]), 'spearman': float(np.corrcoef(rk(E), rk(costs))[0, 1]),
            'couplings': int((np.abs(np.triu(Q, 1)) > 1e-9).sum())}


def surrogate_table(scs):
    out = {}
    for rule in RULES:
        rows = [surrogate_row(s, rule) for s in scs]
        reg = np.array([r['regret'] for r in rows])
        out[rule] = {'mean_regret': float(reg.mean()), 'max_regret': float(reg.max()),
                     'frac_le_5pct': float((reg <= 0.05).mean()), 'frac_zero': float((reg == 0).mean()),
                     'regret_points_of_10': float(np.mean(np.clip((0.25 - reg) / 0.20, 0, 1)) * 10),
                     'mean_pearson': float(np.mean([r['pearson'] for r in rows])),
                     'min_pearson': float(np.min([r['pearson'] for r in rows])),
                     'mean_spearman': float(np.mean([r['spearman'] for r in rows])),
                     'mean_couplings': float(np.mean([r['couplings'] for r in rows])),
                     'R_regret': rows[0]['regret'], 'R_pearson': rows[0]['pearson'], 'rows': rows}
    return out


def seeds_job(args):
    rule, seed = args
    return qaoa_eval.job((tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json'), seed, rule))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qaoa', action='store_true')
    ap.add_argument('--seeds', action='store_true')
    a = ap.parse_args()
    res_dir = HERE / 'results'
    res_dir.mkdir(exist_ok=True)

    w = closed_form.check(n=100)
    print(f'closed-form pairwise physics vs brute-force finite differences (100 scenarios): max err {w:.1e}')
    R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
    _, _, info = aq.build_analytic_qubo(R, return_info=True)
    print(f"single-approach 4-slot recursions per analytic QUBO: {info['approach_sims']} "
          f"(A-NS 16 + A-EW 16 + unlinked B 16 + linked B 8 platoons x 16 = 128); joint 8-bit schedules simulated: 0")

    scs = qaoa_eval.scenario_set('val')
    sur = surrogate_table(scs)
    (res_dir / 'surrogate_val.json').write_text(json.dumps(sur, indent=1))
    print('\nSurrogate quality on VAL (R + 60 scenarios seed 555)')
    print(f"{'rule':<12} {'mean reg':>9} {'max reg':>8} {'<=5%':>6} {'=0':>6} {'pts/10':>7} {'Pearson':>8} {'Spearman':>9} {'#J':>5} {'R reg':>6}")
    for r, d in sur.items():
        print(f"{r:<12} {d['mean_regret']*100:8.2f}% {d['max_regret']*100:7.1f}% {d['frac_le_5pct']:6.3f} {d['frac_zero']:6.3f} "
              f"{d['regret_points_of_10']:7.2f} {d['mean_pearson']:8.4f} {d['mean_spearman']:9.4f} {d['mean_couplings']:5.1f} "
              f"{d['R_regret']*100:5.1f}%")

    if a.qaoa:
        jobs = [(s, 0, r) for r in ('adaptive', 'webster') for s in scs]
        with ProcessPoolExecutor(2) as ex:
            rows = list(ex.map(qaoa_eval.job, jobs, chunksize=2))
        print('\nFull QAOA pipeline on VAL (p=4 split/qubit, Gibbs eta=2, 300 evals, 4096 shots, seed 0)')
        for r in ('adaptive', 'webster'):
            agg = qaoa_eval.aggregate([x for x in rows if x['rule'] == r])
            (res_dir / f'qaoa_val_{r}_seed0.json').write_text(
                json.dumps({'summary': agg, 'rows': [x for x in rows if x['rule'] == r]}, indent=1))
            print(f"{r:<10} R lift {agg['R_lift']:.4f} R enr {agg['R_enrichment']:.2f} | mean lift {agg['mean_lift']:.4f} "
                  f"min lift {agg['min_lift']:.4f} mean enr {agg['mean_enrichment']:.2f} P(opt) {agg['mean_p_optimal']:.3f} "
                  f"regret<=5% {agg['frac_regret_le_5pct']:.3f}")
    if a.seeds:
        jobs = [(r, sd) for r in ('adaptive', 'webster') for sd in range(5)]
        with ProcessPoolExecutor(2) as ex:
            rows = list(ex.map(seeds_job, jobs))
        (res_dir / 'R_seeds.json').write_text(json.dumps(rows, indent=1))
        print('\nScenario R, sampling seeds 0-4')
        for r in ('adaptive', 'webster'):
            rr = [x for x in rows if x['rule'] == r]
            print(f"{r:<10} lift min {min(x['lift'] for x in rr):.4f} mean {np.mean([x['lift'] for x in rr]):.4f} | "
                  f"enr mean {np.mean([x['enrichment'] for x in rr]):.2f} | P(opt) mean {np.mean([x['p_optimal'] for x in rr]):.3f}")


if __name__ == '__main__':
    main()
