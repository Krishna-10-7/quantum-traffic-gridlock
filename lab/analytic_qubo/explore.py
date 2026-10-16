"""Exploration harness (tuning set only: random_scenarios(n, seed=31)).

Compares several local / closed-form QUBO rules against the repo's projection QUBO
on surrogate metrics (regret, Pearson, Gibbs-proxy lift). Not the final script.
"""
import os
import sys
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import itertools  # noqa: E402
import numpy as np  # noqa: E402

import qubo as qb  # noqa: E402
import traffic_model as tm  # noqa: E402
import baselines as bl  # noqa: E402

X = qb.X


# ------------------------------------------------------------- single approach
def phi(arr, green, c, S, w, extra=(0, 0, 0, 0)):
    q = 0
    cost = 0.0
    deps = []
    for t in range(4):
        avail = q + arr[t] + extra[t]
        d = min(avail, c) if green[t] else 0
        q = avail - d
        cost += q + w * max(0, q - S)
        deps.append(d)
    return cost, deps


def components(s):
    """List of (vars, F) where F maps a dict {var: bit} -> cost; sum over components + switching = official cost."""
    c, S, w = s['service_capacity'], s['storage_capacity'], s['overflow_weight']
    lp, f = s['link']['phase'], s['link']['fraction']
    up = 'NS' if lp == 'EW' else 'EW'
    G = lambda phase, b: [int(v) if phase == 'NS' else 1 - int(v) for v in b]
    arr = s['arrivals']
    comps = []

    def mk_A(phase):
        return lambda xa: phi(arr['A'][phase], G(phase, xa), c, S, w)[0]

    def B_unl(xb):
        return phi(arr['B'][up], G(up, xb), c, S, w)[0]

    def B_link(xa3, xb):
        _, d = phi(arr['A'][lp], G(lp, list(xa3) + [0]), c, S, w)
        tau = [0] + [tm.transfer(f, d[t]) for t in range(3)]
        return phi(arr['B'][lp], G(lp, xb), c, S, w, tau)[0]

    comps.append(('A_NS', [0, 1, 2, 3], lambda x: mk_A('NS')([x[i] for i in range(4)])))
    comps.append(('A_EW', [0, 1, 2, 3], lambda x: mk_A('EW')([x[i] for i in range(4)])))
    comps.append(('B_unl', [4, 5, 6, 7], lambda x: B_unl([x[i] for i in range(4, 8)])))
    comps.append(('B_link', [0, 1, 2, 4, 5, 6, 7], lambda x: B_link([x[i] for i in range(3)], [x[i] for i in range(4, 8)])))
    return comps


def red_anchor(s, name):
    """Own all-red anchor of a component, in x space (dict var->bit)."""
    lp = s['link']['phase']
    up = 'NS' if lp == 'EW' else 'EW'
    red = lambda phase: 0 if phase == 'NS' else 1   # x value that makes `phase` red
    if name == 'A_NS':
        return {i: 0 for i in range(4)}
    if name == 'A_EW':
        return {i: 1 for i in range(4)}
    if name == 'B_unl':
        return {i: red(up) for i in range(4, 8)}
    return {**{i: red(lp) for i in range(3)}, **{i: red(lp) for i in range(4, 8)}}


def mobius_add(F, vars_, anchor, Q, off_acc):
    """Second-order Moebius (finite-difference) expansion of F around `anchor`, added to Q (x space)."""
    base = dict(anchor)
    f0 = F(base)

    def flip(*idx):
        x = dict(base)
        for i in idx:
            x[i] = 1 - x[i]
        return F(x)
    f1 = {i: flip(i) - f0 for i in vars_}
    f2 = {(i, j): flip(i, j) - f1[i] - f1[j] - f0 for i, j in itertools.combinations(vars_, 2)}
    # y_i = sig_i x_i + rho_i
    sig = {i: (1 if base[i] == 0 else -1) for i in vars_}
    rho = {i: (0 if base[i] == 0 else 1) for i in vars_}
    off = f0
    for i in vars_:
        off += f1[i] * rho[i]
        Q[i, i] += f1[i] * sig[i]
    for (i, j), v in f2.items():
        off += v * rho[i] * rho[j]
        Q[i, i] += v * sig[i] * rho[j]
        Q[j, j] += v * rho[i] * sig[j]
        Q[min(i, j), max(i, j)] += v * sig[i] * sig[j]
    off_acc[0] += off
    return 1 + len(vars_) + len(f2)


def switching_add(s, Q, off_acc):
    w = s['switch_weight']
    for n in (0, 4):
        for t in range(1, 4):
            i, j = n + t - 1, n + t
            Q[i, i] += w
            Q[j, j] += w
            Q[i, j] += -2 * w


def build_mobius(s, anchor_mode='red', ref=None):
    Q = np.zeros((8, 8))
    off = [0.0]
    nsim = 0
    for name, vars_, F in components(s):
        if anchor_mode == 'red':
            anc = red_anchor(s, name)
        else:
            anc = {i: int(ref[i]) for i in vars_}
        nsim += mobius_add(F, vars_, anc, Q, off)
    switching_add(s, Q, off)
    return off[0], Q, nsim


def comp_tables(s):
    comps = components(s)
    tabs = []
    for name, vars_, F in comps:
        v = np.array([F({i: int(row[i]) for i in range(8)}) for row in tm.ALL_BITS])
        tabs.append((name, vars_, v))
    return tabs


def build_local_ls(s, weights_fn=None):
    """Least squares per component on its own local patterns (vars only)."""
    Q = np.zeros((8, 8))
    off = 0.0
    for name, vars_, F in components(s):
        k = len(vars_)
        pats = np.array(list(itertools.product([0, 1], repeat=k)))
        vals = np.array([F({v: int(p[m]) for m, v in enumerate(vars_)}) for p in pats])
        pairs = list(itertools.combinations(range(k), 2))
        feats = np.column_stack([np.ones(len(pats))] + [pats[:, m] for m in range(k)] + [pats[:, a] * pats[:, b] for a, b in pairs])
        wts = np.ones(len(pats)) if weights_fn is None else weights_fn(name, vars_, pats)
        sw = np.sqrt(wts)
        coef, *_ = np.linalg.lstsq(feats * sw[:, None], vals * sw, rcond=None)
        off += coef[0]
        for m, v in enumerate(vars_):
            Q[v, v] += coef[1 + m]
        for n_, (a, b) in enumerate(pairs):
            i, j = vars_[a], vars_[b]
            Q[min(i, j), max(i, j)] += coef[1 + k + n_]
    off_l = [off]
    switching_add(s, Q, off_l)
    return off_l[0], Q


# ------------------------------------------------------------- metrics
def gibbs_metrics(E, costs, eta=2.0):
    w = np.exp(-eta * (E - E.min()) / (E.std() or 1))
    p = w / w.sum()
    return bl.distribution_metrics(costs, p)


def surrogate_metrics(E, costs):
    k = int(np.argmin(E))
    cstar = costs.min()
    reg = (costs[k] - cstar) / cstar if cstar > 0 else 0.0
    g = gibbs_metrics(E, costs)
    rk = lambda v: np.argsort(np.argsort(v))
    return {'regret': reg, 'pearson': float(np.corrcoef(E, costs)[0, 1]),
            'spearman': float(np.corrcoef(rk(E), rk(costs))[0, 1]),
            'g_lift': g['lift'], 'g_enr': g['enrichment'], 'maxerr': float(np.abs(E - costs).max())}


def summarize(rows, label):
    reg = np.array([r['regret'] for r in rows])
    out = {'label': label, 'mean_regret': reg.mean(), 'max_regret': reg.max(), 'le5': (reg <= 0.05).mean(),
           'zero': (reg == 0).mean(), 'pts10': float(np.mean(np.clip((0.25 - reg) / 0.20, 0, 1)) * 10)}
    for k in ('pearson', 'spearman', 'g_lift', 'g_enr'):
        out[k] = float(np.mean([r[k] for r in rows]))
    out['min_g_lift'] = float(np.min([r['g_lift'] for r in rows]))
    return out


def fmt(d):
    return (f"{d['label']:<34} regret mean {d['mean_regret']*100:5.2f}% max {d['max_regret']*100:5.1f}% "
            f"<=5%: {d['le5']:.3f} zero: {d['zero']:.3f} pts {d['pts10']:.2f} | r {d['pearson']:.4f} "
            f"rho {d['spearman']:.4f} | Gibbs lift {d['g_lift']:.4f} (min {d['min_g_lift']:.3f}) enr {d['g_enr']:.2f}")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    scs = tm.random_scenarios(n, seed=31)
    variants = {}
    for s in scs:
        costs = tm.cost_table(s)
        # sanity: decomposition
        tabs = comp_tables(s)
        Qs = np.zeros((8, 8)); o = [0.0]; switching_add(s, Qs, o)
        tot = sum(v for _, _, v in tabs) + qb.energies(Qs, 0.0)
        assert np.allclose(tot, costs), 'decomposition mismatch'
        cands = {}
        for meas in ('uniform', 'adaptive'):
            off, Q = qb.build_qubo(s, measure=meas)
            cands['proj_' + meas] = qb.energies(Q, off)
        off, Q, _ = build_mobius(s, 'red')
        cands['mobius_red'] = qb.energies(Q, off)
        for nm, ref in (('ref_0', '00000000'), ('ref_1', '11111111'), ('ref_greedy', bl.greedy(s)['bits'])):
            off, Q, _ = build_mobius(s, 'ref', ref)
            cands['mobius_' + nm] = qb.energies(Q, off)
        off, Q = build_local_ls(s)
        cands['local_ls_uniform'] = qb.energies(Q, off)
        for k, E in cands.items():
            variants.setdefault(k, []).append(surrogate_metrics(E, costs))
    for k, rows in variants.items():
        print(fmt(summarize(rows, k)))
