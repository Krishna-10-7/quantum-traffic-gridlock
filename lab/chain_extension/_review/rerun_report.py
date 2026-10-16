"""Reviewer rerun of the headline report set (R3 + 20 chains, generator seed 1001, seeds 0-4) and selected
ablations, with metrics recomputed independently from the raw 4096-shot counts.

  python lab/chain_extension/_review/rerun_report.py report
  python lab/chain_extension/_review/rerun_report.py ablate
  python lab/chain_extension/_review/rerun_report.py subproblem
  python lab/chain_extension/_review/rerun_report.py valmarg
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
EXT = HERE.parent
ROOT = EXT.parents[1]
sys.path[:0] = [str(EXT), str(ROOT)]

import numpy as np  # noqa: E402

import chain_model as cm  # noqa: E402
import chain_run as cr  # noqa: E402
import chain_scorer as cs  # noqa: E402
import traffic_model as tm  # noqa: E402


def my_metrics(costs, counts, elite_size):
    shots = sum(counts.values())
    idx = np.array([int(b, 2) for b in counts])
    n = np.array(list(counts.values()), float)
    c_star, c_rand = costs.min(), costs.mean()
    eq = (costs[idx] * n).sum() / shots
    order = sorted(range(len(costs)), key=lambda i: (costs[i], i))   # ties by ascending bitstring
    elite = set(order[:elite_size])
    fe = sum(c for i, c in zip(idx, n) if i in elite) / shots
    return {'shots': shots, 'lift': float((c_rand - eq) / (c_rand - c_star)),
            'enrichment': float(fe / (elite_size / len(costs))),
            'p_opt': float(n[costs[idx] == c_star].sum() / shots)}


def job(args):
    s, seed, cfg = args
    t = time.time()
    res = cr.solve_chain(s, seed, cfg)
    wall = time.time() - t
    costs = cm.cost_table(s)
    e = np.array(cr.cqb.energies(np.array(res['qubo']['Q_upper_triangular']), res['qubo']['offset']))
    qmin = int(np.argmin(e))
    mine = my_metrics(costs, res['counts'], 208)
    m = res['metrics']
    q = m['baselines']['6_qaoa_raw_samples']
    return {'scenario': s['name'], 'seed': seed, **mine,
            'agent_lift': q['lift'], 'agent_enrichment': q['enrichment'], 'agent_p_opt': q['p_optimal'],
            'regret': float((costs[qmin] - costs.min()) / costs.min()),
            'evals': m['training']['objective_evaluations'], 'qubits': m['circuit']['qubits'],
            'rzz': m['circuit']['ops'].get('rzz', 0), 'n_params': m['circuit']['n_parameters'], 'wall': wall,
            'hybrid': m['baselines']['7_hybrid_labelled']['cost'], 'decoded': m['decoded']['cost'], 'C_star': float(costs.min())}


def pmap(jobs):
    with ProcessPoolExecutor(2) as ex:
        return list(ex.map(job, jobs))


def report_set():
    R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
    return [cm.scenario_R3(R)] + cm.random_chain_scenarios(20, seed=1001)


def summ(rows):
    rnd = [r for r in rows if r['scenario'] != 'R3']
    per = {}
    for r in rows:
        per.setdefault(r['scenario'], []).append(r['lift'])
    return {'n': len(rows),
            'random_mean_lift': float(np.mean([r['lift'] for r in rnd])) if rnd else None,
            'random_min_lift': float(np.min([r['lift'] for r in rnd])) if rnd else None,
            'random_mean_enrich': float(np.mean([r['enrichment'] for r in rnd])) if rnd else None,
            'random_mean_popt': float(np.mean([r['p_opt'] for r in rnd])) if rnd else None,
            'R3_mean_lift': float(np.mean([r['lift'] for r in rows if r['scenario'] == 'R3'])),
            'all_mean_lift': float(np.mean([r['lift'] for r in rows])),
            'regret_le_5_frac_scen': float(np.mean([r['regret'] <= 0.05 for r in rnd if r['seed'] == rnd[0]['seed']])),
            'max_regret': float(max(r['regret'] for r in rows)),
            'worst_seed_min_lift': float(min(min(v) for v in per.values())),
            'max_abs_diff_vs_agent_metrics': float(max(max(abs(r['lift'] - r['agent_lift']),
                                                           abs(r['enrichment'] - r['agent_enrichment']),
                                                           abs(r['p_opt'] - r['agent_p_opt'])) for r in rows)),
            'evals_set': sorted({r['evals'] for r in rows}), 'shots_set': sorted({r['shots'] for r in rows}),
            'mean_wall': float(np.mean([r['wall'] for r in rows])), 'max_wall': float(max(r['wall'] for r in rows)),
            'mean_rzz': float(np.mean([r['rzz'] for r in rows]))}


def report():
    scen = report_set()
    rows = pmap([(s, seed, {}) for s in scen for seed in range(5)])
    rep = json.loads((EXT / 'results' / 'report.json').read_text())
    old = {(r['scenario'], r['seed']): r for r in rep['runs']}
    d = max(abs(r['lift'] - old[(r['scenario'], r['seed'])]['lift']) for r in rows)
    out = {'summary': summ(rows), 'max_abs_lift_diff_vs_agent_report_json': d, 'rows': rows}
    (HERE / 'rerun_report.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['summary'], indent=1), 'diff vs report.json', d)


VARIANTS = {'standard ansatz': {'cost_mode': 'single', 'mixer_mode': 'single'},
            'adaptive, light-cone sparse': {'sparsity': 'lightcone'},
            'kappa=0.5': {'kappa': 0.5},
            'uniform mu (local)': {'measure': 'uniform', 'build': 'local'},
            'p=1': {'p': 1}}


def ablate():
    scen = report_set()
    out = {}
    for name, cfg in VARIANTS.items():
        rows = pmap([(s, 0, cfg) for s in scen])
        a = {'mean_lift': float(np.mean([r['lift'] for r in rows])), 'min_lift': float(min(r['lift'] for r in rows)),
             'mean_enrich': float(np.mean([r['enrichment'] for r in rows])),
             'regret_le_5': float(np.mean([r['regret'] <= 0.05 for r in rows])),
             'mean_rzz': float(np.mean([r['rzz'] for r in rows])), 'evals': sorted({r['evals'] for r in rows})}
        out[name] = a
        print(name, a, flush=True)
    (HERE / 'rerun_ablate.json').write_text(json.dumps(out, indent=1))


def _sub(s):
    import run
    r8 = run.solve(cs.restrict_AB(s), 0)['metrics']
    r12 = cr.solve_chain(s, 0)['metrics']
    return {'scenario': s['name'], 'lift8': r8['baselines']['6_qaoa_raw_samples']['lift'],
            'popt8': r8['baselines']['6_qaoa_raw_samples']['p_optimal'],
            'evals8': r8['training']['objective_evaluations'],
            'lift12': r12['baselines']['6_qaoa_raw_samples']['lift'],
            'popt12': r12['baselines']['6_qaoa_raw_samples']['p_optimal']}


def subproblem():
    scen = cm.random_chain_scenarios(12, seed=31)
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(_sub, scen))
    s = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'scenario'}
    (HERE / 'rerun_subproblem.json').write_text(json.dumps({'mean': s, 'rows': rows}, indent=1))
    print(json.dumps(s, indent=1))


def _valmarg(s2):
    """Embed an official VAL scenario as a chain with an EMPTY C and f2 = 0, run the 12-qubit pipeline, marginalise
    the 4096 shots onto (A,B), and score them with the official 2-node costs; compare to run.solve on the same s2."""
    import run
    import baselines as bl
    r8 = run.solve(s2, 0)['metrics']['baselines']['6_qaoa_raw_samples']
    s3 = cs.embed_two_node(s2, bc_phase='NS', bc_fraction=0.0, name=s2.get('name'))
    res = cr.solve_chain(s3, 0)
    marg = {}
    for b, c in res['counts'].items():
        marg[b[:8]] = marg.get(b[:8], 0) + c
    m = bl.sample_metrics(tm.cost_table(s2), marg)
    c_bits_trivial = sum(c for b, c in res['counts'].items() if b[8:] in ('0000', '1111')) / 4096
    return {'scenario': s2.get('name'), 'lift8': r8['lift'], 'enr8': r8['enrichment'], 'lift12_marg': m['lift'],
            'enr12_marg': m['enrichment'], 'frac_C_no_switch': c_bits_trivial,
            'chain_lift': res['metrics']['baselines']['6_qaoa_raw_samples']['lift']}


def valmarg():
    R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
    VAL = [R] + tm.random_scenarios(60, seed=555)
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(_valmarg, VAL))
    s = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'scenario'}
    s['min_lift8'] = float(min(r['lift8'] for r in rows))
    s['min_lift12_marg'] = float(min(r['lift12_marg'] for r in rows))
    (HERE / 'rerun_valmarg.json').write_text(json.dumps({'mean': s, 'rows': rows}, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == '__main__':
    {'report': report, 'ablate': ablate, 'subproblem': subproblem, 'valmarg': valmarg}[sys.argv[1]]()
