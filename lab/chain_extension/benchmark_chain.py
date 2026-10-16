"""Chain-extension experiments.

  python lab/chain_extension/benchmark_chain.py tune      # design checks on the TUNING set (chain generator seed 31)
  python lab/chain_extension/benchmark_chain.py report    # headline: R3 + 20 chain scenarios (generator seed 1001), seeds 0-4
  python lab/chain_extension/benchmark_chain.py ablate    # one-change ablations on the report set (seed 0)

Report-set numbers are 4,096-shot sampled metrics (StatevectorSampler seed = run seed), chain scorer costs.
Elite set = 208 lowest-cost schedules of 4096 (random enrichment = 1, max 19.69).
Nothing is selected on the report set: the frozen 2-node design (run.DEFAULTS) is used unchanged.
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
sys.path[:0] = [str(HERE), str(ROOT)]

import numpy as np  # noqa: E402

import chain_model as cm  # noqa: E402
import chain_run as cr  # noqa: E402
import chain_scorer as cs  # noqa: E402
import traffic_model as tm  # noqa: E402

RES = HERE / 'results'
WORKERS = 2
TUNE_SEED, REPORT_SEED = 31, 1001


def sequential_decomposition(s, costs):
    """Classical myopic decomposition baseline: choose (A,B) to minimise the A,B cost alone (= official 2-node cost
    plus nothing from C), then choose C optimally given that (A,B). Shows the value of joint coordination."""
    cube = costs.reshape(256, 16)
    ab_only = cm.approach_tables(s)
    ab_cost = sum(v for k, v in ab_only.items() if k != 'switch' and k[0] in 'AB').reshape(256, 16)[:, 0]
    sw_ab = np.array([sum(b[n * 4 + t] != b[n * 4 + t - 1] for n in range(2) for t in range(1, 4)) for b in tm.ALL_BITS])
    ab = int(np.argmin(ab_cost + s['switch_weight'] * sw_ab))
    c = int(np.argmin(cube[ab]))
    return float(cube[ab, c])


def _job(args):
    s, seed, cfg, tag = args
    res = cr.solve_chain(s, seed, cfg)
    d = cr.summarize(res)
    d['tag'] = tag
    d['aligned'] = s['links'][0]['phase'] == s['links'][1]['phase']
    d['sequential'] = sequential_decomposition(s, cm.cost_table(s))
    return d


def pmap(jobs):
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(_job, jobs))


def agg(rows):
    L = np.array([r['lift'] for r in rows])
    E = np.array([r['enrichment'] for r in rows])
    return {'n': len(rows), 'mean_lift': float(L.mean()), 'min_lift': float(L.min()), 'mean_enrichment': float(E.mean()),
            'min_enrichment': float(E.min()), 'mean_exact_lift': float(np.mean([r['exact_lift'] for r in rows])),
            'mean_p_opt': float(np.mean([r['p_optimal'] for r in rows])),
            'frac_regret_le_5pct': float(np.mean([r['regret_rel'] <= 0.05 for r in rows])),
            'mean_regret': float(np.mean([r['regret_rel'] for r in rows])),
            'max_regret': float(np.max([r['regret_rel'] for r in rows])),
            'mean_rzz': float(np.mean([r['rzz'] for r in rows])),
            'mean_seconds': float(np.mean([r['seconds'] for r in rows])),
            'max_seconds': float(np.max([r['seconds'] for r in rows]))}


TUNE_VARIANTS = {
    'final(transferred)': {},
    'standard ansatz': {'cost_mode': 'single', 'mixer_mode': 'single'},
    'split gamma, single beta': {'mixer_mode': 'single'},
    'per-qubit beta, single gamma': {'cost_mode': 'single'},
    'p=1': {'p': 1},
    'p=2': {'p': 2},
    'p=3': {'p': 3},
    'uniform mu (local)': {'measure': 'uniform', 'build': 'local'},
    'adaptive, light-cone sparse': {'sparsity': 'lightcone'},
    'kappa=0.5': {'kappa': 0.5},
    'kappa=2': {'kappa': 2.0},
    'eta=1': {'eta': 1.0},
    'eta=4': {'eta': 4.0},
    'objective <E>': {'kind': 'energy'},
    'stage1=20': {'stage1_evals': 20},
    'stage1=80': {'stage1_evals': 80},
    'init random': {'init': 'random'},
    'kappa=0.5, light-cone sparse': {'kappa': 0.5, 'sparsity': 'lightcone'},
}


def tune(n, names=None, outname='tune_seed31.json'):
    scen = cm.random_chain_scenarios(n, seed=TUNE_SEED)
    out = {}
    for name in names or list(TUNE_VARIANTS):
        cfg = TUNE_VARIANTS[name]
        t = time.time()
        rows = pmap([(s, 0, cfg, name) for s in scen])
        out[name] = {'cfg': cfg, 'summary': agg(rows), 'rows': rows}
        a = out[name]['summary']
        print(f"{name:32s} lift {a['mean_lift']:.3f} (min {a['min_lift']:.3f})  enr {a['mean_enrichment']:.2f}  "
              f"exact-lift {a['mean_exact_lift']:.3f}  regret<=5% {a['frac_regret_le_5pct']:.2f}  rzz {a['mean_rzz']:.0f}  "
              f"[{time.time() - t:.0f}s]", flush=True)
    RES.mkdir(exist_ok=True)
    (RES / outname).write_text(json.dumps(out, indent=1))


def report_set(n=20):
    R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
    return [cm.scenario_R3(R)] + cm.random_chain_scenarios(n, seed=REPORT_SEED)


def report(n, seeds, variant='final(transferred)', outname='report.json'):
    scen = report_set(n)
    sc_dir = HERE / 'scenarios'
    sc_dir.mkdir(exist_ok=True)
    for s in scen:
        (sc_dir / f"{s['name']}.json").write_text(json.dumps(s, indent=1))
    t = time.time()
    rows = pmap([(s, seed, TUNE_VARIANTS[variant], variant) for s in scen for seed in seeds])
    by = {}
    for r in rows:
        by.setdefault(r['scenario'], []).append(r)
    per = []
    for name, rs in by.items():
        r0 = rs[0]
        per.append({'scenario': name, 'aligned': r0['aligned'], 'C_star': r0['C_star'], 'C_rand': r0['C_rand'],
                    'alternating': r0['alternating'], 'greedy': r0['greedy'], 'sequential': r0['sequential'],
                    'qubo_min': r0['qubo_min'], 'regret_rel': r0['regret_rel'], 'pearson': r0['pearson'],
                    'mean_E_Q': float(np.mean([r['E_Q'] for r in rs])),
                    'decoded': [r['decoded'] for r in rs], 'hybrid': [r['hybrid'] for r in rs],
                    'lift_mean': float(np.mean([r['lift'] for r in rs])), 'lift_min': float(np.min([r['lift'] for r in rs])),
                    'enrichment_mean': float(np.mean([r['enrichment'] for r in rs])),
                    'enrichment_min': float(np.min([r['enrichment'] for r in rs])),
                    'p_opt_mean': float(np.mean([r['p_optimal'] for r in rs])),
                    'exact_lift': r0['exact_lift'], 'rzz': r0['rzz'], 'depth': r0['depth'], 'evals': r0['evals'],
                    'seconds_max': float(np.max([r['seconds'] for r in rs]))})
    gap = lambda c, r: (r['C_rand'] - c) / (r['C_rand'] - r['C_star'])
    rand_sc = [p for p in per if p['scenario'] != 'R3']
    summary = {
        'all_runs': agg(rows),
        'random_scenarios_only': agg([r for r in rows if r['scenario'] != 'R3']),
        'R3': agg([r for r in rows if r['scenario'] == 'R3']),
        'robustness_min_lift_over_seeds': {'mean_over_scenarios': float(np.mean([p['lift_min'] for p in per])),
                                           'min_over_scenarios': float(np.min([p['lift_min'] for p in per]))},
        'gap_closed_mean_random': {k: float(np.mean([gap(p[k] if not isinstance(p[k], list) else np.mean(p[k]), p)
                                                     for p in rand_sc]))
                                   for k in ('alternating', 'greedy', 'sequential', 'qubo_min', 'mean_E_Q', 'decoded', 'hybrid')},
        'frac_decoded_optimal': float(np.mean([d == p['C_star'] for p in per for d in p['decoded']])),
        'frac_hybrid_optimal': float(np.mean([d == p['C_star'] for p in per for d in p['hybrid']])),
        'aligned_vs_not': {str(a): {'n': len([p for p in rand_sc if p['aligned'] == a]),
                                    'mean_lift': float(np.mean([p['lift_mean'] for p in rand_sc if p['aligned'] == a])),
                                    'mean_regret': float(np.mean([p['regret_rel'] for p in rand_sc if p['aligned'] == a])),
                                    'mean_rzz': float(np.mean([p['rzz'] for p in rand_sc if p['aligned'] == a]))}
                           for a in (True, False) if any(p['aligned'] == a for p in rand_sc)},
        'wall_seconds': time.time() - t,
    }
    RES.mkdir(exist_ok=True)
    (RES / outname).write_text(json.dumps({'variant': variant, 'cfg': TUNE_VARIANTS[variant], 'summary': summary,
                                           'per_scenario': per, 'runs': rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for p in per:
        print(f"{p['scenario']:14s} al={int(p['aligned'])} C*={p['C_star']:7.2f} Crand={p['C_rand']:7.2f} "
              f"alt={p['alternating']:7.2f} gr={p['greedy']:7.2f} seq={p['sequential']:7.2f} qmin={p['qubo_min']:7.2f} "
              f"EQ={p['mean_E_Q']:7.2f} lift={p['lift_mean']:.3f}/{p['lift_min']:.3f} enr={p['enrichment_mean']:5.2f} "
              f"popt={p['p_opt_mean']:.3f} reg={p['regret_rel']:.3f} t={p['seconds_max']:.1f}s")


ABLATE_VARIANTS = ['final(transferred)', 'standard ansatz', 'split gamma, single beta', 'p=2', 'uniform mu (local)',
                   'adaptive, light-cone sparse']


def ablate(n, extra):
    scen = report_set(n)
    names = ABLATE_VARIANTS + [e for e in extra if e not in ABLATE_VARIANTS]
    out = {}
    for name in names:
        cfg = TUNE_VARIANTS[name]
        rows = pmap([(s, 0, cfg, name) for s in scen])
        out[name] = {'cfg': cfg, 'summary': agg(rows), 'rows': rows}
        a = out[name]['summary']
        print(f"{name:32s} lift {a['mean_lift']:.3f} (min {a['min_lift']:.3f})  enr {a['mean_enrichment']:.2f}  "
              f"P(opt) {a['mean_p_opt']:.3f}  regret {a['mean_regret']:.4f} (<=5%: {a['frac_regret_le_5pct']:.2f})  "
              f"rzz {a['mean_rzz']:.0f}", flush=True)
    RES.mkdir(exist_ok=True)
    (RES / 'ablations_report_set.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['tune', 'report', 'ablate'])
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--variants', nargs='*', default=None, help='tune: subset of TUNE_VARIANTS')
    ap.add_argument('--variant', default='final(transferred)', help='report: which TUNE_VARIANTS entry')
    ap.add_argument('--outname', default=None)
    a = ap.parse_args()
    if a.cmd == 'tune':
        tune(a.n or 12, a.variants, a.outname or 'tune_seed31.json')
    elif a.cmd == 'report':
        report(a.n or 20, list(range(a.seeds)), a.variant, a.outname or 'report.json')
    else:
        ablate(a.n or 20, a.extra)
