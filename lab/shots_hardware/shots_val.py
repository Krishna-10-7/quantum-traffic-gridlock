"""Headline numbers for shot-based training on VAL = [R] + random_scenarios(60, seed=555).

Every training objective evaluation is an N-shot estimate (seeded multinomial from the exact circuit
distribution); <= 300 evaluations including probes; final metrics from exactly 4096 shots of the
final bound circuit (Qiskit StatevectorSampler, seed = run seed), i.e. the same metric as run.solve.

    python lab/shots_hardware/shots_val.py     -> results/shots_val.json, results/shots_val.md
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np  # noqa: E402

import common as C  # noqa: E402

# frozen after tuning on random_scenarios(20, seed=31) (results/tune_*.jsonl)
SPSA = {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'spsa_A': 10, 'spsa_avg': 0.25, 'select': 'last'}
NAIVE = {'method': 'cobyla'}          # the repo optimiser unchanged, fed noisy estimates
VARIANTS = [('exact (current final config)', None, {}),
            ('naive COBYLA, N=256', 256, NAIVE),
            ('SPSA, N=128', 128, SPSA), ('SPSA, N=256', 256, SPSA), ('SPSA, N=1024', 1024, SPSA),
            ('SPSA, exact objective (N=1e8)', 10 ** 8, SPSA)]
VAL = C.val_set()


def job(args):
    si, seed, N, tcfg = args
    if N is None:   # current final config, exactly as the repo computes it
        r = C.run.solve(VAL[si], seed)['metrics']
        q = r['baselines']['6_qaoa_raw_samples']
        return {'si': si, 'seed': seed, 'evals': r['training']['objective_evaluations'],
                'sampled': {k: q[k] for k in ('E_Q', 'lift', 'enrichment', 'p_optimal')},
                'exact': r['qaoa_exact_distribution'],
                'regret_rel': r['surrogate']['regret_rel']}
    r = C.solve_shots(VAL[si], seed, N, tcfg)
    r['si'] = si
    return r


def agg(rows):
    L = np.array([r['sampled']['lift'] for r in rows])
    En = np.array([r['sampled']['enrichment'] for r in rows])
    Le = np.array([r['exact']['lift'] for r in rows])
    return {'mean_lift': float(L.mean()), 'min_lift': float(L.min()), 'mean_enrichment': float(En.mean()),
            'min_enrichment': float(En.min()), 'mean_p_opt': float(np.mean([r['sampled']['p_optimal'] for r in rows])),
            'mean_lift_exact_dist': float(Le.mean()), 'max_evals': int(max(r['evals'] for r in rows))}


def main():
    t0 = time.time()
    out = {'val': {}, 'R_seeds': {}, 'config_spsa': SPSA}
    with ProcessPoolExecutor(2) as ex:
        for name, N, tcfg in VARIANTS:
            rows = list(ex.map(job, [(i, 0, N, tcfg) for i in range(len(VAL))], chunksize=2))
            rseeds = list(ex.map(job, [(0, sd, N, tcfg) for sd in range(5)]))
            a = agg(rows)
            a['R_lift_seed0'] = rows[0]['sampled']['lift']
            a['R_enrichment_seed0'] = rows[0]['sampled']['enrichment']
            out['val'][name] = {'summary': a, 'rows': rows}
            rl = [r['sampled']['lift'] for r in rseeds]
            re_ = [r['sampled']['enrichment'] for r in rseeds]
            out['R_seeds'][name] = {'lift': rl, 'enrichment': re_, 'min_lift': min(rl), 'mean_lift': float(np.mean(rl)),
                                    'mean_enrichment': float(np.mean(re_))}
            print(f"{name:32s} VAL lift {a['mean_lift']:.3f} (min {a['min_lift']:.3f}) enr {a['mean_enrichment']:.2f} "
                  f"P(opt) {a['mean_p_opt']:.3f} | R 5 seeds lift min {min(rl):.3f} mean {np.mean(rl):.3f} "
                  f"enr {np.mean(re_):.2f}  [{time.time() - t0:.0f}s]", flush=True)
    cur = out['val']['exact (current final config)']['rows']
    regs = np.array([r['regret_rel'] for r in cur])
    out['frac_regret_le_5pct'] = float(np.mean(regs <= 0.05 + 1e-12))
    (C.HERE / 'results').mkdir(exist_ok=True)
    (C.HERE / 'results' / 'shots_val.json').write_text(json.dumps(out, indent=1, default=float))
    lines = ['# Shot-based training on VAL (R + 60 scenarios, generator seed 555)', '',
             'Final metrics: 4096 shots of the final bound circuit, sampling seed 0 (as run.solve). '
             'R robustness: seeds 0-4 (seed drives training shots AND final shots).', '',
             '| training | VAL mean lift | VAL min lift | VAL mean enrich. | VAL min enrich. | P(opt) | '
             'R lift (seed 0) | R min lift (5 seeds) | R mean lift | R mean enrich. |', '|' + '---|' * 10]
    for name, _, _ in VARIANTS:
        a = out['val'][name]['summary']
        rs = out['R_seeds'][name]
        lines.append(f"| {name} | {a['mean_lift']:.3f} | {a['min_lift']:.3f} | {a['mean_enrichment']:.2f} | "
                     f"{a['min_enrichment']:.2f} | {a['mean_p_opt']:.3f} | {a['R_lift_seed0']:.3f} | {rs['min_lift']:.3f} | "
                     f"{rs['mean_lift']:.3f} | {rs['mean_enrichment']:.2f} |")
    lines += ['', f"Surrogate regret <= 5% on {out['frac_regret_le_5pct']:.1%} of VAL (same QUBO for all rows)."]
    (C.HERE / 'results' / 'shots_val.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
