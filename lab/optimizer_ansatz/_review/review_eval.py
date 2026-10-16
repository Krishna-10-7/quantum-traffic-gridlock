"""Independent reviewer re-run of the optimizer_ansatz claims (does not import the agent's harness).

    python lab/optimizer_ansatz/_review/review_eval.py val      # VAL: current vs candidate vs no-coupling, R seeds 0-4
    python lab/optimizer_ansatz/_review/review_eval.py g2026    # extra held-out: random_scenarios(100, seed=2026)

* VAL is regenerated from traffic_model (not read from the agent's cache) and compared with the cache.
* Candidate = exactly the proposed integration: run.solve with qaoa.train replaced by qaoa_nft.train and
  cfg {'cost_mode': 'field', 'stage1_evals': 10}.
* During training a guard makes scorer.evaluate / traffic_model.cost_table / approach_costs / approach_trace /
  cost raise, and every Ansatz.statevector call is counted.
* An instrumented copy of qaoa_nft.train counts memo hits (cached repeated-parameter calls).
* Metrics are recomputed from the counts with an independent implementation over scorer.evaluate.
"""
import os
import sys

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import json
import pickle
import time
import types
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
ROOT = AGENT.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import qaoa  # noqa: E402
import run  # noqa: E402
import scorer  # noqa: E402
import traffic_model as tm  # noqa: E402

ORIG_TRAIN = qaoa.train
STATE = {'training': False, 'sims': 0, 'memo_hits': 0}
_PATCHED = {'done': False}


def load_nft(instrumented):
    src = (AGENT / 'qaoa_nft.py').read_text()
    if instrumented:
        old = "            if memo['x'] is not None and np.array_equal(x, memo['x']):\n                return memo['v']"
        assert old in src
        src = src.replace(old, "            if memo['x'] is not None and np.array_equal(x, memo['x']):\n"
                               "                _REVIEW_STATE['memo_hits'] += 1\n                return memo['v']")
    mod = types.ModuleType('qaoa_nft_review' if instrumented else 'qaoa_nft_orig')
    mod.__dict__['_REVIEW_STATE'] = STATE
    exec(compile(src, str(AGENT / 'qaoa_nft.py'), 'exec'), mod.__dict__)
    return mod


def patch_once():
    if _PATCHED['done']:
        return
    _PATCHED['done'] = True
    orig_sv = qaoa.Ansatz.statevector

    def sv(self, params):
        if STATE['training']:
            STATE['sims'] += 1
        return orig_sv(self, params)
    qaoa.Ansatz.statevector = sv

    def guard(mod, name):
        f = getattr(mod, name)

        def g(*a, **k):
            if STATE['training']:
                raise RuntimeError(f'official-cost function {mod.__name__}.{name} called during training')
            return f(*a, **k)
        setattr(mod, name, g)
    guard(scorer, 'evaluate')
    for n in ('cost_table', 'approach_costs', 'approach_trace', 'cost'):
        guard(tm, n)


def make_train(nft, no_coupling=False):
    def f(h, J, energies, **kw):
        if no_coupling:
            J = np.zeros_like(J)
        STATE.update(training=True, sims=0, memo_hits=0)
        try:
            a, x, hist = nft.train(h, J, energies, **kw)
        finally:
            STATE['training'] = False
        f.last = {'sims': STATE['sims'], 'memo_hits': STATE['memo_hits'], 'hist': len(hist)}
        return a, x, hist
    f.last = None
    return f


def counted_orig_train(h, J, energies, **kw):
    STATE.update(training=True, sims=0, memo_hits=0)
    try:
        out = ORIG_TRAIN(h, J, energies, **kw)
    finally:
        STATE['training'] = False
    counted_orig_train.last = {'sims': STATE['sims'], 'memo_hits': 0, 'hist': len(out[2])}
    return out


CAND_CFG = {'cost_mode': 'field', 'stage1_evals': 10}


def my_metrics(s, counts):
    costs = np.array([scorer.evaluate(s, [int(c) for c in b])['cost'] for b in tm.BITSTRINGS])
    bits = tm.BITSTRINGS
    shots = sum(counts.values())
    cs, cr = costs.min(), costs.mean()
    eq = sum(costs[int(b, 2)] * n for b, n in counts.items()) / shots
    elite = sorted(range(256), key=lambda i: (costs[i], bits[i]))[:13]
    eset = {bits[i] for i in elite}
    fe = sum(n for b, n in counts.items() if b in eset) / shots
    popt = sum(n for b, n in counts.items() if costs[int(b, 2)] == cs) / shots
    return costs, {'lift': (cr - eq) / (cr - cs), 'enrichment': fe / (13 / 256), 'p_optimal': popt}


def single_qubit_entropy(a, x):
    psi = a.statevector(x).reshape((2,) * 8)
    out = []
    for i in range(8):
        m = np.moveaxis(psi, i, 0).reshape(2, -1)
        ev = np.clip(np.linalg.eigvalsh(m @ m.conj().T).real, 1e-300, 1)
        out.append(float(-(ev * np.log2(ev)).sum()))
    return float(np.mean(out))


def job(args):
    arm, s, seed = args
    patch_once()
    if arm == 'current':
        qaoa.train = counted_orig_train
        tr, cfg = counted_orig_train, None
    else:
        nft = load_nft(instrumented=True)
        tr = make_train(nft, no_coupling=(arm == 'nocoup'))
        qaoa.train = tr
        cfg = CAND_CFG
    try:
        t = time.time()
        res = run.solve(s, seed, cfg)
        wall = time.time() - t
    finally:
        qaoa.train = ORIG_TRAIN
    m = res['metrics']
    q = m['baselines']['6_qaoa_raw_samples']
    costs, mine = my_metrics(s, res['counts'])
    # final-state objective vs reported min(history)
    h = np.array(res['qubo']['ising']['h'])
    J = np.zeros((8, 8)) if arm == 'nocoup' else np.array(res['qubo']['ising']['J_upper'])
    c = m['config']
    a = qaoa.Ansatz(h, J, c['p'], c['cost_mode'], c['mixer_mode'])
    x = np.array(res['training']['params'])
    off = res['qubo']['offset']
    Q = np.array(res['qubo']['Q_upper_triangular'])
    import qubo as qb
    E = qb.energies(Q, off)
    f_final = qaoa.make_objective(a, E, 'gibbs', 0.25, 2.0)(x)
    qmin = int(np.argmin(E))
    ceiling_lift = (costs.mean() - costs[qmin]) / (costs.mean() - costs.min())
    pr = a.probabilities(x)
    return {'arm': arm, 'scenario': s.get('name'), 'seed': seed,
            'lift': q['lift'], 'enrichment': q['enrichment'], 'p_optimal': q['p_optimal'],
            'my_lift': mine['lift'], 'my_enrichment': mine['enrichment'], 'my_p_optimal': mine['p_optimal'],
            'shots': sum(res['counts'].values()), 'qubits': m['circuit']['qubits'], 'p': m['circuit']['p'],
            'n_params': m['circuit']['n_parameters'], 'rzz': int(m['circuit']['ops'].get('rzz', 0)),
            'consistency': m['circuit']['max_abs_prob_diff_circuit_vs_trainer'],
            'evals_reported': m['training']['objective_evaluations'], 'train_info': tr.last,
            'min_history': m['training']['final_objective'], 'f_final': float(f_final),
            'regret_rel': m['surrogate']['regret_rel'], 'qubo_ceiling_lift': float(ceiling_lift),
            'p_qubo_min': float(pr[qmin]), 'entropy1q': single_qubit_entropy(a, x),
            'max_abs_param': float(np.abs(x).max()), 'seconds': m['runtime_seconds']['total'], 'wall': wall,
            'counts_hash': __import__('hashlib').sha256(json.dumps(sorted(res['counts'].items())).encode()).hexdigest(), 'params': res['training']['params']}


def summarize(rows):
    L = np.array([r['lift'] for r in rows])
    En = np.array([r['enrichment'] for r in rows])
    R = [r for r in rows if r['scenario'] == 'R']
    ti = [r['train_info'] for r in rows]
    return {'n': len(rows), 'R_lift': R[0]['lift'] if R else None, 'R_enr': R[0]['enrichment'] if R else None,
            'R_popt': R[0]['p_optimal'] if R else None,
            'mean_lift': float(L.mean()), 'min_lift': float(L.min()), 'mean_enr': float(En.mean()),
            'min_enr': float(En.min()), 'mean_popt': float(np.mean([r['p_optimal'] for r in rows])),
            'frac_regret_le_5': float(np.mean([r['regret_rel'] <= 0.05 for r in rows])),
            'mean_ceiling_lift': float(np.mean([r['qubo_ceiling_lift'] for r in rows])),
            'mean_p_qubo_min': float(np.mean([r['p_qubo_min'] for r in rows])),
            'max_metric_diff_vs_mine': float(max(max(abs(r['lift'] - r['my_lift']), abs(r['enrichment'] - r['my_enrichment']),
                                                    abs(r['p_optimal'] - r['my_p_optimal'])) for r in rows)),
            'all_4096': all(r['shots'] == 4096 for r in rows), 'max_p': max(r['p'] for r in rows),
            'max_qubits': max(r['qubits'] for r in rows),
            'sims_eq_hist': all(t['sims'] == t['hist'] for t in ti), 'max_sims': max(t['sims'] for t in ti),
            'min_sims': min(t['sims'] for t in ti), 'max_memo_hits': max(t['memo_hits'] for t in ti),
            'max_sims_plus_memo': max(t['sims'] + t['memo_hits'] for t in ti),
            'max_consistency': float(max(r['consistency'] for r in rows)),
            'mean_entropy1q': float(np.mean([r['entropy1q'] for r in rows])),
            'max_abs_param': float(max(r['max_abs_param'] for r in rows)),
            'mean_final_minus_minhist': float(np.mean([r['f_final'] - r['min_history'] for r in rows])),
            'max_final_minus_minhist': float(max(r['f_final'] - r['min_history'] for r in rows)),
            'mean_seconds': float(np.mean([r['seconds'] for r in rows])),
            'max_seconds': float(max(r['seconds'] for r in rows))}


def val_scenarios():
    Rs = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
    return [Rs] + tm.random_scenarios(60, seed=555)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'val'
    out = {}
    if which == 'val':
        scs = val_scenarios()
        cache = pickle.loads((AGENT / 'cache' / 'val_R_s555_n60.pkl').read_bytes())
        out['cache_matches_fresh_VAL'] = (len(cache) == len(scs) and all(c[0] == s for c, s in zip(cache, scs)))
        print('cache matches fresh VAL:', out['cache_matches_fresh_VAL'], flush=True)
        arms = ['current', 'candidate', 'nocoup']
    else:
        scs = tm.random_scenarios(100, seed=2026)
        arms = ['current', 'candidate']
    with Pool(2) as pool:
        for arm in arms:
            t = time.time()
            rows = pool.map(job, [(arm, s, 0) for s in scs])
            out[arm] = {'summary': summarize(rows), 'rows': [{k: v for k, v in r.items() if k != 'params'} for r in rows]}
            print(arm, json.dumps(out[arm]['summary']), f'{time.time() - t:.0f}s', flush=True)
        if which == 'val':
            for arm in ('current', 'candidate'):
                rr = pool.map(job, [(arm, scs[0], sd) for sd in range(5)])
                out[arm]['R_seeds'] = {'lifts': [r['lift'] for r in rr], 'enr': [r['enrichment'] for r in rr],
                                       'min_lift_clipped': float(np.clip(min(r['lift'] for r in rr), 0, 1))}
                print(arm, 'R seeds', json.dumps(out[arm]['R_seeds']), flush=True)
            # determinism: candidate twice on R and 3 VAL scenarios
            d1 = pool.map(job, [('candidate', s, 3) for s in scs[:4]])
            d2 = pool.map(job, [('candidate', s, 3) for s in scs[:4]])
            out['determinism'] = all(a['params'] == b['params'] and a['counts_hash'] == b['counts_hash']
                                     for a, b in zip(d1, d2))
            print('determinism', out['determinism'], flush=True)
    (HERE / f'review_{which}.json').write_text(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
