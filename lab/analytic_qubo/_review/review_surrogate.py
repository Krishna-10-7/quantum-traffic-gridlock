"""Independent reviewer checks of the analytic_qubo track (surrogate level, no QAOA).

Writes only to lab/analytic_qubo/_review/.
"""
import itertools
import json
import os
import sys
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
ROOT = AGENT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENT))

import numpy as np  # noqa: E402

import qubo as qb  # noqa: E402
import scorer  # noqa: E402
import traffic_model as tm  # noqa: E402
import analytic_qubo as aq  # noqa: E402
import closed_form  # noqa: E402


def val_set():
    return [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(60, seed=555)


# ---------------------------------------------------------------- 1. builder never touches joint cost / scorer
class Forbidden(Exception):
    pass


def guarded_build(s, **kw):
    saved = (tm.cost_table, tm.approach_costs, tm.cost, scorer.evaluate, qb.build_qubo, qb.approach_tables)

    def boom(*a, **k):
        raise Forbidden('forbidden call')
    tm.cost_table = tm.approach_costs = tm.cost = scorer.evaluate = qb.build_qubo = qb.approach_tables = boom
    try:
        return aq.build_analytic_qubo(s, **kw)
    finally:
        (tm.cost_table, tm.approach_costs, tm.cost, scorer.evaluate, qb.build_qubo, qb.approach_tables) = saved


# ---------------------------------------------------------------- 2. math claim: Taylor == L2(Bern(p*)) projection
def product_projection(s, p8):
    """My own weighted LS on the joint 256 table with product weights Bernoulli(p*) (switching included)."""
    costs = tm.cost_table(s)
    X = tm.ALL_BITS.astype(float)
    w = np.prod(np.where(X == 1, p8, 1 - p8), axis=1)
    F = qb.FEATURES
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(F * sw[:, None], costs * sw, rcond=None)
    off, Q = qb.coef_to_qubo(coef)
    return off, Q


def my_regret(E, costs):
    k = int(np.argmin(E))
    return float((costs[k] - costs.min()) / costs.min())


def main():
    out = {}
    scs = val_set()
    costs_all = [tm.cost_table(s) for s in scs]

    # 1. guarded build
    for s in scs:
        for rule in ('webster', 'uniform', 'red', 'local_gibbs'):
            guarded_build(s, rule=rule)
    out['guarded_build_ok'] = True

    # 1b. local decomposition matches scorer on extra scenarios (not VAL, seed 4242, no gap filter)
    extra = tm.random_scenarios(150, seed=4242, min_gap=0.0)
    worst = 0.0
    X = tm.ALL_BITS
    for s in extra:
        comps = aq.local_components(s)
        tot = np.zeros(256)
        for name, vars_, pats, vals in comps:
            idx = X[:, vars_] @ (2 ** np.arange(len(vars_) - 1, -1, -1))
            tot += vals[idx]
        tot += s['switch_weight'] * sum((X[:, n + t] != X[:, n + t - 1]) for n in (0, 4) for t in range(1, 4))
        worst = max(worst, float(np.abs(tot - tm.cost_table(s)).max()))
    out['local_decomp_vs_scorer_max_err_150_extra'] = worst

    # 2. Taylor == product-measure projection; uniform == repo uniform
    err_t, err_u = 0.0, 0.0
    for s, c in zip(scs, costs_all):
        o, Q = aq.build_analytic_qubo(s)
        o2, Q2 = product_projection(s, aq.webster_point(s))
        err_t = max(err_t, float(np.abs(qb.energies(Q, o) - qb.energies(Q2, o2)).max()))
        o, Q = aq.build_analytic_qubo(s, rule='uniform')
        o2, Q2 = qb.build_qubo(s, measure='uniform')
        err_u = max(err_u, float(np.abs(qb.energies(Q, o) - qb.energies(Q2, o2)).max()))
    out['taylor_vs_my_product_projection_max_energy_err'] = err_t
    out['uniform_rule_vs_repo_uniform_max_energy_err'] = err_u

    # 3. surrogate metrics on VAL (my own computation)
    builders = {
        'adaptive': lambda s: qb.build_qubo(s, measure='adaptive', kappa=1.0),
        'webster': lambda s: aq.build_analytic_qubo(s),
        'uniform': lambda s: qb.build_qubo(s, measure='uniform'),
        'red': lambda s: aq.build_analytic_qubo(s, rule='red'),
        'local_gibbs': lambda s: aq.build_analytic_qubo(s, rule='local_gibbs'),
    }
    sur = {}
    per = {}
    for name, b in builders.items():
        reg, pr, ncoup, a4b = [], [], [], []
        for s, c in zip(scs, costs_all):
            o, Q = b(s)
            E = qb.energies(Q, o)
            reg.append(my_regret(E, c))
            pr.append(float(np.corrcoef(E, c)[0, 1]))
            ncoup.append(int((np.triu(Q, 1) != 0).sum()))
            a4b.append(int((Q[3, 4:] != 0).sum()))
        reg = np.array(reg)
        per[name] = reg
        sur[name] = {'frac_le_5pct': float((reg <= 0.05).mean()), 'frac_zero': float((reg == 0).mean()),
                     'max_regret': float(reg.max()), 'mean_regret': float(reg.mean()),
                     'points10': float(np.mean(np.clip((0.25 - reg) / 0.20, 0, 1)) * 10),
                     'mean_pearson': float(np.mean(pr)), 'mean_couplings': float(np.mean(ncoup)),
                     'R_couplings': ncoup[0], 'mean_A4B_nonzero': float(np.mean(a4b)), 'R_regret': float(reg[0])}
    out['surrogate_VAL'] = sur
    d = per['webster'] - per['adaptive']
    out['webster_minus_adaptive_regret'] = {'webster_higher': int((d > 1e-12).sum()), 'webster_lower': int((d < -1e-12).sum()),
                                            'max_webster_regret_where_higher': float(per['webster'][d > 1e-12].max())
                                            if (d > 1e-12).any() else None}

    # 4. coefficient agreement
    IU = np.triu_indices(8, 1)
    vec = lambda o, Q: np.r_[qb.qubo_to_ising(Q, o)[0], qb.qubo_to_ising(Q, o)[1][IU]]
    prs, l2s, a4b_rel = [], [], []
    for s in scs:
        va, vd = vec(*aq.build_analytic_qubo(s)), vec(*qb.build_qubo(s, measure='adaptive'))
        prs.append(np.corrcoef(va, vd)[0, 1])
        l2s.append(np.linalg.norm(va - vd) / np.linalg.norm(vd))
        Jd = qb.qubo_to_ising(*reversed(qb.build_qubo(s, measure='adaptive')))[1]
        a4b_rel.append(np.abs(Jd[3, 4:]).max() / np.abs(Jd).max())
    out['coef'] = {'mean_pearson': float(np.mean(prs)), 'min_pearson': float(np.min(prs)), 'mean_rel_l2': float(np.mean(l2s)),
                   'adaptive_A4B_over_maxJ_max': float(np.max(a4b_rel))}

    # 4b. pruning check: drop adaptive A4-B couplings (what a 96-RZZ adaptive circuit would look like)
    reg_pruned = []
    for s, c in zip(scs, costs_all):
        o, Q = qb.build_qubo(s, measure='adaptive')
        Q = Q.copy()
        Q[3, 4:] = 0.0
        reg_pruned.append(my_regret(qb.energies(Q, o), c))
    reg_pruned = np.array(reg_pruned)
    out['adaptive_pruned_A4B'] = {'frac_le_5pct': float((reg_pruned <= 0.05).mean()), 'max_regret': float(reg_pruned.max()),
                                  'same_minimizer_regret_as_unpruned': bool(np.allclose(reg_pruned, per['adaptive']))}

    # 5. recursion count
    _, _, info = aq.build_analytic_qubo(scs[0], return_info=True)
    out['approach_sims_reported'] = info['approach_sims']
    # actual number of approach() calls (without the counter decrement)
    calls = {'n': 0}
    orig = aq.approach

    def counting(*a, **k):
        calls['n'] += 1
        return orig(*a, **k)
    aq.approach = counting
    aq.local_components(scs[0])
    aq.approach = orig
    out['approach_calls_actual'] = calls['n']

    # 6. closed form check
    out['closed_form_max_err_100'] = float(closed_form.check(n=100))

    # 7. Webster constants: inspect defaults
    import inspect
    out['webster_point_signature'] = str(inspect.signature(aq.webster_point))
    out['build_signature'] = str(inspect.signature(aq.build_analytic_qubo))

    print(json.dumps(out, indent=1))
    (HERE / 'review_surrogate.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
