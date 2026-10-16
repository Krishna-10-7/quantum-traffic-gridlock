"""Reproducible experiments for the report.

  python experiments/benchmark.py seeds --scenario scenarios/scenario_R.json --out outputs/R
  python experiments/benchmark.py generalization --n 100 --gen-seed 2026
  python experiments/benchmark.py ablations --n 40 --gen-seed 99

All numbers are official-scorer costs of the 4,096 raw final-circuit shots unless stated.
Random scenarios are drawn from the hidden-scenario ranges in the problem statement
(traffic_model.random_scenario) and filtered to C_rand - C* > 5.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

import run  # noqa: E402
import traffic_model as tm  # noqa: E402

RESULTS = ROOT / 'results'
WORKERS = max(1, min(6, (os.cpu_count() or 2) - 1))


def summarize(res):
    b = res['metrics']['baselines']
    q = b['6_qaoa_raw_samples']
    return {
        'scenario': res['metrics']['scenario'], 'seed': res['metrics']['seed'],
        'C_star': q['C_star'], 'C_rand': q['C_rand'], 'E_Q': q['E_Q'],
        'lift': q['lift'], 'enrichment': q['enrichment'], 'p_optimal': q['p_optimal'],
        'alternating': b['1_best_alternating']['cost'], 'greedy': b['2_greedy_larger_queue']['cost'],
        'qubo_min': b['5_qubo_exact_minimizer']['cost'], 'qaoa_schedule': res['schedule']['official_cost'],
        'qaoa_best_sample': q['best_sampled']['cost'], 'hybrid': b['7_hybrid_labelled']['cost'],
        'regret_rel': res['metrics']['surrogate']['regret_rel'],
        'pearson': res['metrics']['surrogate']['pearson_qubo_vs_official'],
        'evals': res['metrics']['training']['objective_evaluations'],
        'seconds': res['metrics']['runtime_seconds']['total'],
        'exact_lift': res['metrics']['qaoa_exact_distribution']['lift'],
        'exact_enrichment': res['metrics']['qaoa_exact_distribution']['enrichment'],
    }


def _job(args):
    s, seed, cfg = args
    return summarize(run.solve(s, seed, cfg))


def pmap(jobs):
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(_job, jobs))


def stats(rows, key):
    v = np.array([r[key] for r in rows], float)
    return {'mean': float(v.mean()), 'median': float(np.median(v)), 'min': float(v.min()), 'max': float(v.max())}


# ------------------------------------------------------------------ seeds
def cmd_seeds(a):
    s = tm.load_scenario(a.scenario)
    out = Path(a.out)
    rows = []
    for seed in a.seeds:
        res = run.solve(s, seed)
        d = out / f'seed{seed}'
        d.mkdir(parents=True, exist_ok=True)
        (d / 'qubo.json').write_text(json.dumps(res['qubo'], indent=2))
        (d / 'circuit.qasm').write_text(res['qasm'])
        (d / 'counts.json').write_text(json.dumps(res['counts'], indent=1))
        (d / 'schedule.json').write_text(json.dumps(res['schedule'], indent=2))
        (d / 'metrics.json').write_text(json.dumps(res['metrics'], indent=2))
        (d / 'training.json').write_text(json.dumps(res['training']))
        rows.append(summarize(res))
    lifts = [r['lift'] for r in rows]
    summary = {'scenario': s.get('name'), 'seeds': a.seeds, 'rows': rows,
               'min_lift': min(lifts), 'min_lift_clipped': float(np.clip(min(lifts), 0, 1)),
               'mean_lift': float(np.mean(lifts)), 'mean_enrichment': float(np.mean([r['enrichment'] for r in rows]))}
    (out / 'robustness.json').write_text(json.dumps(summary, indent=2))
    lines = [f"# Seed robustness - scenario {s.get('name')}", '',
             f"C* = {rows[0]['C_star']}, C_rand = {rows[0]['C_rand']:.3f}", '',
             '| seed | E_Q | lift | enrichment | P(opt) | schedule cost | best sample | hybrid |',
             '|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['E_Q']:.3f} | {r['lift']:.4f} | {r['enrichment']:.2f} | {r['p_optimal']:.3f} "
                     f"| {r['qaoa_schedule']} | {r['qaoa_best_sample']} | {r['hybrid']} |")
    lines += ['', f"**min lift (clipped) = {summary['min_lift_clipped']:.4f}**, mean lift = {summary['mean_lift']:.4f}, "
                  f"mean enrichment = {summary['mean_enrichment']:.2f}",
              '', '| baseline | official cost |', '|---|---|',
              f"| (1) best alternating | {rows[0]['alternating']} |",
              f"| (2) greedy larger-queue | {rows[0]['greedy']} |",
              f"| (3) exact enumeration C* | {rows[0]['C_star']} |",
              f"| (4) uniform random expected | {rows[0]['C_rand']:.3f} |",
              f"| (5) QUBO exact minimizer (surrogate ceiling) | {rows[0]['qubo_min']} |",
              f"| (6) QAOA raw samples, mean over 4096 shots (seed 0) | {rows[0]['E_Q']:.3f} |",
              f"| (6) QAOA decoded schedule (lowest QUBO energy sample) | {rows[0]['qaoa_schedule']} |",
              f"| (7) HYBRID: QAOA samples + 1-flip descent | {rows[0]['hybrid']} |"]
    (out / 'robustness.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# ------------------------------------------------------------------ generalization
def cmd_generalization(a):
    scs = tm.random_scenarios(a.n, seed=a.gen_seed)
    t = time.time()
    rows = pmap([(s, 0, None) for s in scs])
    rel = lambda r, k: (r[k] - r['C_star']) / r['C_star'] if r['C_star'] > 0 else 0.0
    gap = lambda r, k: (r['C_rand'] - r[k]) / (r['C_rand'] - r['C_star'])  # "lift" of a deterministic method
    reg = np.array([r['regret_rel'] for r in rows])
    summary = {
        'n': a.n, 'gen_seed': a.gen_seed, 'wall_seconds': time.time() - t,
        'lift': stats(rows, 'lift'), 'enrichment': stats(rows, 'enrichment'), 'p_optimal': stats(rows, 'p_optimal'),
        'surrogate_regret_rel': {**stats(rows, 'regret_rel'), 'frac_le_5pct': float((reg <= 0.05).mean()),
                                 'frac_zero': float((reg == 0).mean()),
                                 'rubric_points_of_10': float(np.mean(np.clip((0.25 - reg) / 0.20, 0, 1)) * 10)},
        'pearson_qubo_vs_official': stats(rows, 'pearson'),
        'gap_closed': {k: float(np.mean([gap(r, k) for r in rows])) for k in
                       ('alternating', 'greedy', 'qubo_min', 'E_Q', 'qaoa_schedule', 'hybrid')},
        'excess_over_optimum_rel': {k: float(np.mean([rel(r, k) for r in rows])) for k in
                                    ('alternating', 'greedy', 'qubo_min', 'qaoa_schedule', 'qaoa_best_sample', 'hybrid')},
        'frac_schedule_optimal': float(np.mean([r['qaoa_schedule'] == r['C_star'] for r in rows])),
        'frac_hybrid_optimal': float(np.mean([r['hybrid'] == r['C_star'] for r in rows])),
        'runtime_seconds': stats(rows, 'seconds'),
        'rows': rows,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'generalization.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))


# ------------------------------------------------------------------ ablations
ABLATIONS = [
    ('final (p=4, split cost, per-qubit mixer, Gibbs, adaptive QUBO)', {}),
    ('depth p=1', {'p': 1}), ('depth p=2', {'p': 2}), ('depth p=3', {'p': 3}),
    ('standard QAOA ansatz (1 gamma, 1 beta per layer)', {'cost_mode': 'single', 'mixer_mode': 'single'}),
    ('per-qubit mixer only (1 gamma per layer)', {'cost_mode': 'single'}),
    ('objective <E> (standard)', {'kind': 'energy'}),
    ('objective CVaR alpha=0.25', {'kind': 'cvar'}),
    ('Gibbs eta=1', {'eta': 1.0}), ('Gibbs eta=4', {'eta': 4.0}),
    ('QUBO: uniform Walsh projection', {'measure': 'uniform'}),
    ('QUBO: adaptive kappa=0.5', {'kappa': 0.5}), ('QUBO: adaptive kappa=2', {'kappa': 2.0}),
    ('init: random angles (3 probes)', {'init': 'random'}),
    ('init: near-identity', {'init': 'small'}),
    ('budget 100 evaluations', {'max_evals': 100}), ('budget 200 evaluations', {'max_evals': 200}),
]


def cmd_ablations(a):
    scs = [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(a.n, seed=a.gen_seed)
    table = []
    for name, cfg in ABLATIONS:
        t = time.time()
        rows = pmap([(s, 0, cfg) for s in scs])
        rnd = rows[1:]
        entry = {'name': name, 'cfg': cfg, 'R_lift': rows[0]['lift'], 'R_enrichment': rows[0]['enrichment'],
                 'lift': stats(rnd, 'lift'), 'enrichment': stats(rnd, 'enrichment'),
                 'p_optimal': stats(rnd, 'p_optimal'),
                 'regret_rel_mean': float(np.mean([r['regret_rel'] for r in rnd])),
                 'frac_regret_le_5pct': float(np.mean([r['regret_rel'] <= 0.05 for r in rnd]))}
        table.append(entry)
        print(f"{name:65s} R {entry['R_lift']:.3f}/{entry['R_enrichment']:5.2f} | lift {entry['lift']['mean']:.3f} "
              f"(min {entry['lift']['min']:.3f}) enr {entry['enrichment']['mean']:5.2f} | {time.time() - t:.0f}s", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'ablations.json').write_text(json.dumps({'n_random': a.n, 'gen_seed': a.gen_seed, 'table': table}, indent=2))
    lines = [f'# Ablations (scenario R + {a.n} random hidden-range scenarios, seed 0, 4096 shots)', '',
             '| variant | R lift | R enrich. | mean lift | min lift | mean enrich. | mean P(opt) | regret<=5% |',
             '|---|---|---|---|---|---|---|---|']
    for e in table:
        lines.append(f"| {e['name']} | {e['R_lift']:.3f} | {e['R_enrichment']:.2f} | {e['lift']['mean']:.3f} | "
                     f"{e['lift']['min']:.3f} | {e['enrichment']['mean']:.2f} | {e['p_optimal']['mean']:.3f} | "
                     f"{e['frac_regret_le_5pct']:.2f} |")
    (RESULTS / 'ablations.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('seeds')
    p.add_argument('--scenario', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p = sub.add_parser('generalization')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--gen-seed', type=int, default=2026)
    p = sub.add_parser('ablations')
    p.add_argument('--n', type=int, default=40)
    p.add_argument('--gen-seed', type=int, default=99)
    a = ap.parse_args()
    {'seeds': cmd_seeds, 'generalization': cmd_generalization, 'ablations': cmd_ablations}[a.cmd](a)


if __name__ == '__main__':
    main()
