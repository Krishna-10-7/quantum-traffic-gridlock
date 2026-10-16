"""Headline numbers on VAL = [scenario R] + random_scenarios(60, seed=555), 4096 sampled shots (seed 0).

    python lab/transfer_init/evaluate_val.py            # candidate vs current + initialisation ablation
                                                        # + scenario R over sampling seeds 0..4

Metric: official-scorer cost of the 4,096 raw shots of the final bound circuit, StatevectorSampler(seed=0),
exactly as run.solve; lift / enrichment / P(opt) per scenario; R reported separately; exact (infinite-shot)
metrics also stored. 'current' is run.solve(s, 0)['metrics'] itself (repo code, unmodified).
Writes results/val.json and results/val.md.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

import transfer_init as TI  # noqa: E402

WORKERS = int(os.environ.get('TI_WORKERS', '2'))
RES = C.HERE / 'results'
RES.mkdir(exist_ok=True)

# initialisation ablation: everything else = candidate settings except the start
VARIANTS = [
    ('candidate: ' + TI.CANDIDATE_NAME, TI.CANDIDATE_CFG),
] + TI.ABLATION_VARIANTS


def _current(args):
    import run
    s, _ = args[:2]
    seed = args[2] if len(args) > 2 else 0
    m = run.solve(s, seed)['metrics']
    q = m['baselines']['6_qaoa_raw_samples']
    return {'name': s.get('name'), 'sampled': {k: q[k] for k in ('E_Q', 'lift', 'enrichment', 'p_optimal')},
            'exact': m['qaoa_exact_distribution'], 'regret_rel': m['surrogate']['regret_rel'],
            'evals': m['training']['objective_evaluations'], 'seconds': m['runtime_seconds']['total']}


def _cand(args):
    s, costs, cfg = args[:3]
    seed = args[3] if len(args) > 3 else 0
    r = C.solve_ti(s, seed, TI.resolve(cfg), costs=costs)
    r.pop('params')
    return r


def pmap(fn, jobs):
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, jobs))


def block(rows):
    R, rnd = rows[0], rows[1:]
    sm = C.summarize(rnd, 'sampled')
    se = C.summarize(rnd, 'exact')
    return {'R_lift': R['sampled']['lift'], 'R_enrichment': R['sampled']['enrichment'], 'R_p_opt': R['sampled']['p_optimal'],
            'random60': sm, 'random60_exact': se, 'all61': C.summarize(rows, 'sampled'),
            'frac_regret_le_5pct': float(np.mean([r['regret_rel'] <= 0.05 for r in rows])),
            'max_evals_used': int(max(r['evals'] for r in rows)),
            'max_seconds': float(max(r['seconds'] for r in rows)),
            'lifts': [r['sampled']['lift'] for r in rows]}


def main():
    data = C.val_set()
    assert data[0][0]['name'] == 'R' and len(data) == 61
    out = {}
    t = time.time()
    rows = pmap(_current, data)
    out['current (run.solve, repo defaults)'] = block(rows)
    print(f"current: {C.fmt(out['current (run.solve, repo defaults)']['all61'])}  ({time.time() - t:.0f}s)", flush=True)
    only = sys.argv[1:]
    for label, cfg in VARIANTS:
        if only and not any(o in label for o in only):
            continue
        t = time.time()
        rows = pmap(_cand, [(s, c, cfg) for s, c in data])
        out[label] = block(rows)
        out[label]['cfg'] = {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in cfg.items()}
        print(f"{label}: {C.fmt(out[label]['all61'])}  ({time.time() - t:.0f}s)", flush=True)
    # scenario R over sampling seeds 0..4 (training is seed-independent for ramp / transfer starts; the random
    # start uses the seed as its init seed, as qaoa.train does)
    R, cR = data[0]
    rob = {}
    rows = pmap(_current, [(R, cR, sd) for sd in range(5)])
    rob['current (run.solve, repo defaults)'] = rows
    for label, cfg in VARIANTS:
        if only and not any(o in label for o in only):
            continue
        c = dict(cfg)
        jobs = []
        for sd in range(5):
            cs = dict(c, init_seed=sd) if c.get('init') == 'random' else c
            jobs.append((R, cR, cs, sd))
        rob[label] = pmap(_cand, jobs)
    out['_R_seeds_0_4'] = {}
    for label, rows in rob.items():
        L = [r['sampled']['lift'] for r in rows]
        En = [r['sampled']['enrichment'] for r in rows]
        out['_R_seeds_0_4'][label] = {'lifts': L, 'enrichments': En, 'min_lift': min(L), 'mean_lift': float(np.mean(L)),
                                      'mean_enrichment': float(np.mean(En)), 'min_enrichment': min(En)}
        print(f"R seeds 0-4 | {label}: lift mean {np.mean(L):.4f} min {min(L):.4f} | enr mean {np.mean(En):.2f} "
              f"min {min(En):.2f}", flush=True)
    (RES / 'val.json').write_text(json.dumps(out, indent=1))
    lines = ['# VAL = scenario R + random_scenarios(60, seed=555); 4096 shots, sampler seed 0', '',
             '| variant | R lift | R enrich. | mean lift (60) | min lift (60) | mean enrich. (60) | min enrich. | '
             'mean P(opt) | exact mean lift | evals max |', '|---|---|---|---|---|---|---|---|---|---|']
    for k, b in out.items():
        if k.startswith('_'):
            continue
        r = b['random60']
        lines.append(f"| {k} | {b['R_lift']:.3f} | {b['R_enrichment']:.2f} | {r['mean_lift']:.4f} | {r['min_lift']:.3f} | "
                     f"{r['mean_enrichment']:.2f} | {r['min_enrichment']:.2f} | {r['mean_p_opt']:.3f} | "
                     f"{b['random60_exact']['mean_lift']:.4f} | {b['max_evals_used']} |")
    (RES / 'val.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
