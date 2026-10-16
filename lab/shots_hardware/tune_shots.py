"""Tune the noise-robust optimiser for shot-based training on the TUNING set only
(traffic_model.random_scenarios(20, seed=31); VAL and R are never used here).

Metric: exact-distribution lift / enrichment of the trained circuit (infinite final shots), mean over
scenarios and training seeds, to separate optimiser quality from final-shot noise.

    python lab/shots_hardware/tune_shots.py [stage]     -> lab/shots_hardware/results/tune_<stage>.jsonl
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np  # noqa: E402

import common as C  # noqa: E402

OUT = C.HERE / 'results'
SC = C.tune_set(20)


def job(args):
    si, seed, N, tcfg = args
    r = C.solve_shots(SC[si], seed, N, tcfg, final_sample=False)
    return r['exact']['lift'], r['exact']['enrichment']


def evaluate(N, tcfg, seeds=(0,), pool=None):
    jobs = [(i, sd, N, tcfg) for i in range(len(SC)) for sd in seeds]
    res = list(pool.map(job, jobs, chunksize=2))
    L = np.array([r[0] for r in res])
    En = np.array([r[1] for r in res])
    return {'N': N, 'cfg': tcfg, 'seeds': list(seeds), 'mean_lift': float(L.mean()), 'min_lift': float(L.min()),
            'p10_lift': float(np.percentile(L, 10)), 'mean_enr': float(En.mean())}


STAGES = {
    'a': [(None, {})] + [(256, c) for c in [
        {'method': 'cobyla'},
        {'method': 'cobyla', 'rhobeg': 0.3, 'rho2': 0.15},
        {'method': 'cobyla', 'rhobeg': 0.3, 'rho2': 0.3},
        {'method': 'cobyla', 'tol': 0.02},
        {'method': 'cobyla', 'rhobeg': 0.3, 'rho2': 0.15, 'tol': 0.02},
        {'method': 'cobyla', 'rhobeg': 0.3, 'rho2': 0.15, 'tol': 0.05},
        {'method': 'cobyla', 'stage1_evals': 100},
        {'method': 'cobyla', 'stage1_evals': 300 - 0},
        {'method': 'cobyla', 'cost_mode': 'single', 'mixer_mode': 'single', 'rhobeg': 0.3},
        {'method': 'cobyla', 'cost_mode': 'split', 'mixer_mode': 'single'},
        {'method': 'cobyla', 'reeval': 20},
        {'method': 'spsa', 'spsa_a': 0.05, 'spsa_c': 0.1},
        {'method': 'spsa', 'spsa_a': 0.1, 'spsa_c': 0.1},
        {'method': 'spsa', 'spsa_a': 0.2, 'spsa_c': 0.1},
        {'method': 'spsa', 'spsa_a': 0.4, 'spsa_c': 0.1},
        {'method': 'spsa', 'spsa_a': 0.2, 'spsa_c': 0.2},
        {'method': 'spsa', 'spsa_a': 0.2, 'spsa_c': 0.05},
        {'method': 'spsa', 'spsa_a': 0.2, 'spsa_c': 0.1, 'stage1_evals': 100},
        {'method': 'spsa', 'spsa_a': 0.2, 'spsa_c': 0.1, 'reeval': 20},
    ]],
    'b': [(256, c) for c in [
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1},
        {'method': 'spsa', 'spsa_a': 0.8, 'spsa_c': 0.1},
        {'method': 'spsa', 'spsa_a': 1.2, 'spsa_c': 0.1},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.15},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'spsa_A': 30},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'spsa_avg': 0.0},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'spsa_avg': 0.5},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'reeval': 20},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'stage1_evals': 20},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'stage1_evals': 3},
    ]],
    'c': [(256, c) for c in [
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'select': 'last'},
        {'method': 'spsa', 'spsa_a': 0.5, 'spsa_c': 0.1},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'spsa_A': 5},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'cost_mode': 'single'},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'mixer_mode': 'single'},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'stage1_evals': 60},
    ]] + [(N, {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1}) for N in (128, 1024, 4096)],
    'd': [(10 ** 8, {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'select': 'last'}),
          (10 ** 8, {'method': 'cobyla'})] + [(256, c) for c in [
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'select': 'last', 'spsa_mom': 0.5},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'select': 'last', 'spsa_mom': 0.8},
        {'method': 'spsa', 'spsa_a': 0.9, 'spsa_c': 0.1, 'select': 'last', 'spsa_mom': 0.8},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'select': 'last', 'spsa_avg': 0.5},
    ]],
    'e': [(N, dict({'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'select': 'last'}, **extra))
          for N in (128, 256) for extra in ({}, {'spsa_clip': 0.1}, {'spsa_clip': 0.2}, {'spsa_c': 0.15, 'spsa_clip': 0.1})],
    # f: does COBYLA (the repo optimiser) recover once each evaluation uses more shots (N <= 4096 = final shots)?
    'f': [(N, c) for N in (1024, 4096) for c in [
        {'method': 'cobyla'},
        {'method': 'cobyla', 'rhobeg': 0.3, 'rho2': 0.3, 'tol': 0.02},
        {'method': 'cobyla', 'reeval': 20},
        {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'select': 'last'},
    ]],
}


SEEDS = {'e': (0, 1)}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'a'
    OUT.mkdir(exist_ok=True)
    path = OUT / f'tune_{stage}.jsonl'
    with ProcessPoolExecutor(int(os.environ.get('TUNE_WORKERS', 2))) as pool, open(path, 'a') as fh:
        for N, cfg in STAGES[stage]:
            t = time.time()
            r = evaluate(N, cfg, seeds=SEEDS.get(stage, (0,)), pool=pool)
            r['sec'] = time.time() - t
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            print(f"N={N} {cfg}  lift {r['mean_lift']:.3f} (min {r['min_lift']:.3f}, p10 {r['p10_lift']:.3f})  "
                  f"enr {r['mean_enr']:.2f}  {r['sec']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
