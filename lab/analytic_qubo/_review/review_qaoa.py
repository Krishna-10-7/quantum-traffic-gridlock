"""Independent reviewer re-run of the full QAOA pipeline: current config vs analytic (Webster) QUBO.

    python lab/analytic_qubo/_review/review_qaoa.py

* current  = run.solve(s, seed) completely unpatched
* webster  = run.solve(s, seed) with qubo.build_qubo replaced by analytic_qubo.build_analytic_qubo(s) (defaults)
Instrumentation (both arms): counts every call of the training objective returned by qaoa.make_objective,
and makes scorer.evaluate / traffic_model.cost_table / approach_costs raise while qaoa.train is running.
Writes only to lab/analytic_qubo/_review/.
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
ROOT = AGENT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT))

import numpy as np  # noqa: E402

import qaoa  # noqa: E402
import qubo as qb  # noqa: E402
import run  # noqa: E402
import scorer  # noqa: E402
import traffic_model as tm  # noqa: E402
import analytic_qubo as aq  # noqa: E402

_ORIG_BUILD = qb.build_qubo
_ORIG_TRAIN = qaoa.train
_ORIG_MKOBJ = qaoa.make_objective
STATE = {'in_train': False, 'obj_calls': 0, 'forbidden_hits': 0}


def _guard(fn):
    def g(*a, **k):
        if STATE['in_train']:
            STATE['forbidden_hits'] += 1
            raise RuntimeError('official cost used during training')
        return fn(*a, **k)
    return g


scorer.evaluate = _guard(scorer.evaluate)
tm.cost_table = _guard(tm.cost_table)
tm.approach_costs = _guard(tm.approach_costs)


def _mkobj(*a, **k):
    f = _ORIG_MKOBJ(*a, **k)

    def counted(x):
        STATE['obj_calls'] += 1
        return f(x)
    return counted


def _train(*a, **k):
    STATE['in_train'] = True
    try:
        return _ORIG_TRAIN(*a, **k)
    finally:
        STATE['in_train'] = False


qaoa.make_objective = _mkobj
qaoa.train = _train


def job(args):
    s, seed, rule = args
    STATE.update(obj_calls=0, forbidden_hits=0)
    if rule == 'webster':
        qb.build_qubo = lambda s_, measure='adaptive', kappa=1.0, **_: aq.build_analytic_qubo(s_)
    else:
        qb.build_qubo = _ORIG_BUILD
    res = run.solve(s, seed)
    qb.build_qubo = _ORIG_BUILD
    m = res['metrics']
    q = m['baselines']['6_qaoa_raw_samples']
    return {'scenario': s.get('name'), 'rule': rule, 'seed': seed, 'lift': q['lift'], 'enrichment': q['enrichment'],
            'p_optimal': q['p_optimal'], 'E_Q': q['E_Q'], 'C_star': q['C_star'], 'C_rand': q['C_rand'],
            'shots': q['shots'], 'regret_rel': m['surrogate']['regret_rel'],
            'exact_lift': m['qaoa_exact_distribution']['lift'],
            'history_len': m['training']['objective_evaluations'], 'obj_calls': STATE['obj_calls'],
            'forbidden_hits': STATE['forbidden_hits'], 'qubits': m['circuit']['qubits'], 'p': m['circuit']['p'],
            'rzz': m['circuit']['ops'].get('rzz', 0), 'seconds': m['runtime_seconds']['total'],
            'counts_hash': hashlib.md5(json.dumps(sorted(res['counts'].items())).encode()).hexdigest()}


def main():
    R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
    val = [R] + tm.random_scenarios(60, seed=555)
    jobs = [(s, 0, r) for r in ('current', 'webster') for s in val]
    jobs += [(R, sd, r) for r in ('current', 'webster') for sd in range(1, 5)]
    jobs += [(R, 0, 'webster'), (R, 0, 'current')]   # determinism repeats
    t = time.time()
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(job, jobs, chunksize=2))
    wall = time.time() - t
    nv = len(val)
    cur, web = rows[:nv], rows[nv:2 * nv]
    seeds = rows[2 * nv:2 * nv + 8]
    rep = rows[-2:]

    def agg(rr):
        reg = np.array([r['regret_rel'] for r in rr])
        return {'mean_lift': float(np.mean([r['lift'] for r in rr])), 'min_lift': float(np.min([r['lift'] for r in rr])),
                'mean_enrichment': float(np.mean([r['enrichment'] for r in rr])),
                'mean_p_optimal': float(np.mean([r['p_optimal'] for r in rr])),
                'frac_regret_le_5pct': float((reg <= 0.05).mean()), 'max_regret': float(reg.max()),
                'R_lift': rr[0]['lift'], 'R_enrichment': rr[0]['enrichment'], 'R_p_optimal': rr[0]['p_optimal'],
                'max_history': int(max(r['history_len'] for r in rr)), 'max_obj_calls': int(max(r['obj_calls'] for r in rr)),
                'history_equals_obj_calls': all(r['history_len'] == r['obj_calls'] for r in rr),
                'forbidden_hits': int(sum(r['forbidden_hits'] for r in rr)),
                'all_4096_shots': all(r['shots'] == 4096 for r in rr), 'qubits': sorted({r['qubits'] for r in rr}),
                'p': sorted({r['p'] for r in rr}), 'R_rzz': rr[0]['rzz'],
                'mean_rzz': float(np.mean([r['rzz'] for r in rr])),
                'max_seconds': float(max(r['seconds'] for r in rr)), 'mean_seconds': float(np.mean([r['seconds'] for r in rr]))}
    out = {'current': agg(cur), 'webster': agg(web)}
    for key in ('lift', 'enrichment', 'p_optimal', 'exact_lift'):
        d = np.array([w[key] - c[key] for w, c in zip(web, cur)])
        out[f'paired_{key}'] = {'mean': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d))),
                                'webster_better': int((d > 0).sum()), 'webster_worse': int((d < 0).sum())}
    # R seeds 0-4
    for r in ('current', 'webster'):
        rr = [cur[0] if r == 'current' else web[0]] + [x for x in seeds if x['rule'] == r]
        out[f'R_seeds_{r}'] = {'lifts': [x['lift'] for x in rr], 'min': float(min(x['lift'] for x in rr)),
                               'mean': float(np.mean([x['lift'] for x in rr])),
                               'enr_mean': float(np.mean([x['enrichment'] for x in rr])),
                               'popt_mean': float(np.mean([x['p_optimal'] for x in rr])),
                               'exact_lift': rr[0]['exact_lift']}
    out['determinism'] = {'webster_R_seed0_repeat_identical': rep[0]['counts_hash'] == web[0]['counts_hash'],
                          'current_R_seed0_repeat_identical': rep[1]['counts_hash'] == cur[0]['counts_hash']}
    out['wall_seconds'] = wall
    # compare with agent's stored rows
    stored = {}
    for r, fn in (('current', 'qaoa_val_adaptive_seed0.json'), ('webster', 'qaoa_val_webster_seed0.json')):
        d = json.loads((AGENT / 'results' / fn).read_text())['rows']
        mine = cur if r == 'current' else web
        stored[r] = {'max_abs_lift_diff_vs_stored': float(max(abs(a['lift'] - b['lift']) for a, b in zip(mine, d))),
                     'max_abs_enr_diff_vs_stored': float(max(abs(a['enrichment'] - b['enrichment']) for a, b in zip(mine, d))),
                     'scenario_order_match': [a['scenario'] for a in mine] == [b['scenario'] for b in d]}
    out['vs_agent_stored_rows'] = stored
    print(json.dumps(out, indent=1))
    (HERE / 'review_qaoa.json').write_text(json.dumps({'summary': out, 'rows': rows}, indent=1, default=str))


if __name__ == '__main__':
    main()
