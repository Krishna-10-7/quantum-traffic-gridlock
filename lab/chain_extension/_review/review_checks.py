"""Reviewer checks (independent of the agent's tests). Writes nothing outside _review/."""
import itertools
import json
import os
import random
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
EXT = HERE.parent
ROOT = EXT.parents[1]
sys.path[:0] = [str(EXT), str(ROOT)]

import numpy as np  # noqa: E402

import chain_model as cm  # noqa: E402
import chain_qaoa as cq  # noqa: E402
import chain_qubo as cqb  # noqa: E402
import chain_run as cr  # noqa: E402
import scorer  # noqa: E402
import traffic_model as tm  # noqa: E402

R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')


# ---------------- 1. independent chain evaluator written from the problem text (not from chain_scorer) -------------
def my_chain_cost(s, bits):
    cap, S = s['service_capacity'], s['storage_capacity']
    l1, l2 = s['links']
    plan = {'A': bits[0:4], 'B': bits[4:8], 'C': bits[8:12]}
    q = {(n, p): 0 for n in 'ABC' for p in ('NS', 'EW')}
    into = {'B': [0] * 5, 'C': [0] * 5}
    delay = over = 0
    for t in range(4):
        for n in 'ABC':
            for p in ('NS', 'EW'):
                a = s['arrivals'][n][p][t]
                if n == 'B' and p == l1['phase']:
                    a += into['B'][t]
                if n == 'C' and p == l2['phase']:
                    a += into['C'][t]
                av = q[(n, p)] + a
                g = (plan[n][t] == 1) == (p == 'NS')
                d = min(av, cap) if g else 0
                q[(n, p)] = av - d
                delay += q[(n, p)]
                over += max(0, q[(n, p)] - S)
                for lk, dst in ((l1, 'B'), (l2, 'C')):
                    if lk['from'] == n and p == lk['phase'] and t < 3:
                        into[dst][t + 1] += int((Decimal(str(lk['fraction'])) * d).quantize(Decimal(1), ROUND_HALF_UP))
    sw = sum(plan[n][t] != plan[n][t - 1] for n in 'ABC' for t in range(1, 4))
    return delay + s['overflow_weight'] * over + s['switch_weight'] * sw


def check_scorer():
    scen = [json.loads(p.read_text()) for p in sorted((EXT / 'scenarios').glob('*.json'))]
    rng = random.Random(123)
    worst = 0.0
    for s in scen:
        tab = cm.cost_table(s)
        for k in rng.sample(range(4096), 400):
            b = [int(c) for c in cm.BITSTRINGS[k]]
            worst = max(worst, abs(my_chain_cost(s, b) - tab[k]))
    # official-scorer-only reduction with a NON-EMPTY C and f2 = 0:
    # C3(a,b,c) = scorer(AB)(a b) + scorer(C as A, B empty, f=0)(c 0000)
    worst_red = 0.0
    for s2 in [R] + tm.random_scenarios(6, seed=4242):
        crng = random.Random(7)
        c_arr = {p: [crng.randint(0, 8) for _ in range(4)] for p in ('NS', 'EW')}
        s3 = {'name': 'x', 'arrivals': {**s2['arrivals'], 'C': c_arr},
              **{k: s2[k] for k in ('service_capacity', 'storage_capacity', 'overflow_weight', 'switch_weight')},
              'links': [dict(s2['link']), {'from': 'B', 'to': 'C', 'phase': 'EW', 'fraction': 0.0}]}
        sC = {'arrivals': {'A': c_arr, 'B': {'NS': [0] * 4, 'EW': [0] * 4}},
              **{k: s2[k] for k in ('service_capacity', 'storage_capacity', 'overflow_weight', 'switch_weight')},
              'link': {'from': 'A', 'to': 'B', 'phase': 'NS', 'fraction': 0.0}}
        tab = cm.cost_table(s3)
        for k in random.Random(9).sample(range(4096), 600):
            bs = cm.BITSTRINGS[k]
            ref = scorer.evaluate(s2, bs[:8])['cost'] + scorer.evaluate(sC, bs[8:] + '0000')['cost']
            worst_red = max(worst_red, abs(ref - tab[k]))
    return {'max_abs_diff_independent_evaluator': worst, 'max_abs_diff_official_reduction_nonempty_C': worst_red}


# ---------------- 2. scenario files == generator ----------------
def check_scenarios():
    gen = [cm.scenario_R3(R)] + cm.random_chain_scenarios(20, seed=1001)
    ok = all(json.loads((EXT / 'scenarios' / f"{s['name']}.json").read_text()) == json.loads(json.dumps(s)) for s in gen)
    # do the generator seeds overlap the forbidden seeds? (seed 31 tune, 1001 report)
    return {'scenario_files_match_generator': ok, 'n': len(gen)}


# ---------------- 3. independent evaluation counter + objective inputs ----------------
def check_budget_and_objective():
    out = []
    calls = {'n': 0}
    orig = cq.Ansatz.probabilities

    def counting(self, params):
        calls['n'] += 1
        return orig(self, params)

    seen = {}
    orig_obj = cq.make_objective

    def spy(ansatz, energies, *a, **k):
        seen.setdefault('energies', []).append(np.array(energies, copy=True))
        return orig_obj(ansatz, energies, *a, **k)

    cq.Ansatz.probabilities = counting
    cq.make_objective = spy
    try:
        for s in [cm.scenario_R3(R)] + cm.random_chain_scenarios(3, seed=1001):
            for cfg in ({}, {'init': 'random'}, {'cost_mode': 'single', 'mixer_mode': 'single'}, {'p': 1}):
                calls['n'] = 0
                seen.clear()
                off, Q, _ = cqb.build_qubo(s)
                e = cqb.energies(Q, off)
                h, J, _ = cqb.qubo_to_ising(Q, off)
                full = {**cr.DEFAULTS, **cfg}
                a, x, hist = cq.train(h, J, e, p=full['p'], cost_mode=full['cost_mode'], mixer_mode=full['mixer_mode'],
                                      kind=full['kind'], eta=full['eta'], max_evals=300, ramp_dts=full['ramp_dts'],
                                      stage1_evals=full['stage1_evals'], init=full['init'], init_seed=0)
                costs = cm.cost_table(s)
                en_ok = all(np.array_equal(v, e) for v in seen['energies'])
                not_cost = all(not np.allclose(v, costs) for v in seen['energies'])
                out.append({'scenario': s['name'], 'cfg': cfg, 'probability_calls_in_train': calls['n'],
                            'len_history': len(hist), 'objective_built_on_qubo_energies': en_ok,
                            'objective_inputs_differ_from_costs': not_cost})
    finally:
        cq.Ansatz.probabilities = orig
        cq.make_objective = orig_obj
    return out


# ---------------- 4. ablation seed-0 rows vs report seed-0 rows ----------------
def check_consistency():
    rep = json.loads((EXT / 'results' / 'report.json').read_text())
    abl = json.loads((EXT / 'results' / 'ablations_report_set.json').read_text())
    r0 = {r['scenario']: r for r in rep['runs'] if r['seed'] == 0}
    a0 = {r['scenario']: r for r in abl['final(transferred)']['rows']}
    keys = ('lift', 'enrichment', 'p_optimal', 'regret_rel', 'E_Q')
    diff = max(abs(r0[k][f] - a0[k][f]) for k in r0 for f in keys)
    return {'max_diff_report_seed0_vs_ablation_final': diff, 'n': len(r0)}


if __name__ == '__main__':
    res = {'scorer': check_scorer(), 'scenarios': check_scenarios(), 'consistency': check_consistency(),
           'budget_objective': check_budget_and_objective()}
    (HERE / 'review_checks.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
