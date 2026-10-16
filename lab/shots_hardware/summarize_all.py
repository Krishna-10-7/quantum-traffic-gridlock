"""Collect the headline numbers of this lab from results/ (no recomputation).

    python lab/shots_hardware/summarize_all.py   -> results/headline.json, results/hardware.md, prints tables

Inputs: results/shots_val.json (shots_val.py), results/hardware.jsonl (hardware.py),
results/e2e_*_summary.json (e2e.py), results/swapnet.json (swapnet.py), results/tune_f.jsonl (tune_shots.py f).
"""
import json
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402

import common as C  # noqa: E402
import hardware as HW  # noqa: E402

R = C.HERE / 'results'


def shots_part(out):
    d = json.loads((R / 'shots_val.json').read_text())
    out['shots_val'] = {}
    for name, v in d['val'].items():
        s = v['summary']
        rs = d['R_seeds'][name]
        out['shots_val'][name] = {'VAL_mean_lift': s['mean_lift'], 'VAL_min_lift': s['min_lift'],
                                  'VAL_mean_enrichment': s['mean_enrichment'], 'VAL_p_opt': s['mean_p_opt'],
                                  'R_lift_seed0': s['R_lift_seed0'], 'R_enrichment_seed0': s['R_enrichment_seed0'],
                                  'R_min_lift_5seeds': rs['min_lift'], 'R_mean_lift_5seeds': rs['mean_lift'],
                                  'R_mean_enrichment_5seeds': rs['mean_enrichment'], 'max_evals': s['max_evals']}
    out['frac_regret_le_5pct'] = d['frac_regret_le_5pct']
    print((R / 'shots_val.md').read_text())


def hardware_part(out):
    rows = [json.loads(line) for line in (R / 'hardware.jsonl').read_text().splitlines()]
    md, agg = HW.summarize(rows)
    complete = {k: v for k, v in agg.items() if v['n'] == len({r['si'] for r in rows})}
    lines = [md, '', '## Best variant per noise level (mean exact-distribution lift over the scenarios)', '',
             '| e2 | best variant | cx heavy-hex | mean lift | min lift | mean enrichment | lift 4096 shots |',
             '|---|---|---|---|---|---|---|']
    best = {}
    for e2 in HW.E2:
        k = max(complete, key=lambda k: complete[k][f'lift_{e2}'])
        d = complete[k]
        best[str(e2)] = k
        lines.append(f"| {e2:g} | {k} | {d['cx_hh']:.0f} | {d[f'lift_{e2}']:.3f} | {d[f'minlift_{e2}']:.3f} | "
                     f"{d[f'enr_{e2}']:.2f} | {d[f'lift4096_{e2}']:.3f} |")
    chk = max(r['routing_check_maxdiff'] for r in rows)
    lines += ['', f'Routing check: max |P_routed - P_ideal| over all {len(rows)} compiled circuits = {chk:.1e}.']
    md = '\n'.join(lines) + '\n'
    (R / 'hardware.md').write_text(md)
    (R / 'hardware_summary.json').write_text(json.dumps(agg, indent=1))
    out['hardware'] = {'variants': agg, 'best_per_e2': best, 'routing_check_maxdiff': chk, 'n_rows': len(rows)}
    print(md)


def e2e_part(out):
    out['e2e'] = {}
    for f in sorted(R.glob('e2e_*_summary.json')):
        d = json.loads(f.read_text())
        out['e2e'][f.stem.replace('_summary', '')] = d
        print(f.stem, 'n =', d['n'], 'cx =', round(d['cx_mean'], 1))
        for arm, s in d['arms'].items():
            print(f'   {arm:32s} lift {s["mean_lift"]:.3f} (min {s["min_lift"]:.3f})  lift4096 {s["mean_lift_4096"]:.3f}  '
                  f'enr4096 {s["mean_enrichment_4096"]:.2f}  R lift4096 {s["R_lift_4096"]:.3f}')


def swapnet_part(out):
    f = R / 'swapnet.json'
    if not f.exists():
        return
    rows = json.loads(f.read_text())
    out['swapnet'] = {}
    for key in sorted({(r['mode'], r['tau'], r['p']) for r in rows}):
        rr = [r for r in rows if (r['mode'], r['tau'], r['p']) == key]
        out['swapnet'][f'{key[0]}{key[1]}_p{key[2]}'] = {
            'n': len(rr), 'sabre_cx': float(np.mean([r['sabre']['n2q'] for r in rr])),
            'swapnet_cx': float(np.mean([r['swapnet']['n2q'] for r in rr])),
            'sabre_depth': float(np.mean([r['sabre']['depth'] for r in rr])),
            'swapnet_depth': float(np.mean([r['swapnet']['depth'] for r in rr]))}
    print((R / 'swapnet.md').read_text())


def tune_f_part(out):
    f = R / 'tune_f.jsonl'
    if not f.exists():
        return
    out['tune_f'] = [json.loads(line) for line in f.read_text().splitlines()]
    print('tuning set (seed 31, 20 scenarios), exact-distribution lift of the trained circuit:')
    for r in out['tune_f']:
        print(f"   N={r['N']:5d} {r['cfg']}  lift {r['mean_lift']:.3f} (min {r['min_lift']:.3f})  enr {r['mean_enr']:.2f}")


def main():
    out = {}
    for part in (shots_part, hardware_part, e2e_part, swapnet_part, tune_f_part):
        part(out)
    (R / 'headline.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
