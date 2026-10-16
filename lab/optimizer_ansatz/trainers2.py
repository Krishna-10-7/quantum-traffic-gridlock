"""Second batch of lab trainers (finishing pass).

train_grow2: layer-wise growth for the multi-angle ansatz that actually trains every layer.

Diagnostic that motivated it: trainers.train_grow restarts its NFT sweep at layer 1 after every growth
step, so with thresholds (150, 200, 250) the 50-evaluation phases (25 coordinates) never reach the newly
appended layers: layers 3 and 4 stay exactly at the identity (all angles 0) and the "p=4" circuit is
functionally a p=1/p=2 circuit ("identity layers: mean 2.00" in results/robust_s7.log and robust_s31.log). train_grow2 fixes this: after a layer is
appended (identity initialisation -> same state, same objective value), the first sweep runs over the
NEW layer's sinusoidal coordinates only, then full sweeps (with COBYLA blocks on the generic coupling
angles) continue until the next threshold.
"""
import numpy as np

import qaoa
import trainers as T


def _layer_of(a, idx):
    """Layer index of flat parameter idx in qaoa.Ansatz a."""
    if idx < a.p * a.mg:
        return idx // a.mg
    return (idx - a.p * a.mg) // a.mb


def train_grow2(h, J, E, cfg=None):
    c = {**T.DEFAULT, **(cfg or {})}
    kind = c['kind']
    budget = T.Budget(c['max_evals'])
    thresholds = list(c['grow']) + [c['max_evals']]
    p = 1
    a = qaoa.Ansatz(h, J, 1, c['cost_mode'], c['mixer_mode'])
    base = qaoa.Ansatz(h, J, 1, 'single', 'single')
    fb = budget.wrap(qaoa.make_objective(base, E, kind, c['alpha'], c['eta']), lambda xx: a.lift_params(xx, base))
    starts = [base.ramp(dt) for dt in c['ramp_dts']]
    vals = [fb(s0) for s0 in starts]
    T._run(fb, starts[int(np.argmin(vals))], 'cobyla', c['stage1_evals'] - budget.n, c['rhobeg'])
    x = budget.best_x.copy()
    budget.best_v = np.inf
    ff = budget.wrap(qaoa.make_objective(a, E, kind, c['alpha'], c['eta']))
    fx = ff(x)
    for stop in thresholds:
        if p > 1:   # new-layer-first sweep
            sidx, freq, _ = T.param_structure(a)
            new = [(j, w) for j, w in zip(sidx, freq) if _layer_of(a, j) == p - 1]
            x, fx = T.nft_sweep(ff, x, fx, [j for j, _ in new], [w for _, w in new], budget, kind, max_n=stop)
            budget.best_v, budget.best_x = fx, x.copy()
        x, fx = T._nft_loop(ff, x, fx, a, budget, kind, c, stop)
        if p == c['p'] or stop >= c['max_evals']:
            break
        g, b = a.split(x)
        a = qaoa.Ansatz(h, J, p + 1, c['cost_mode'], c['mixer_mode'])
        p += 1
        x = a.join(np.vstack([g, np.zeros((1, g.shape[1]))]), np.vstack([b, np.zeros((1, b.shape[1]))]))
        ff = budget.wrap(qaoa.make_objective(a, E, kind, c['alpha'], c['eta']))
        budget.best_v, budget.best_x = fx, x.copy()
    while p < c['p']:
        g, b = a.split(x)
        a = qaoa.Ansatz(h, J, p + 1, c['cost_mode'], c['mixer_mode'])
        p += 1
        x = a.join(np.vstack([g, np.zeros((1, g.shape[1]))]), np.vstack([b, np.zeros((1, b.shape[1]))]))
        budget.best_x = x.copy()
    return a, budget.best_x, budget


def untouched_layers(a, x, tol=0.0):
    """Number of layers whose angles are all exactly zero (identity layers)."""
    g, b = a.split(x)
    return int(sum(np.all(np.abs(g[k]) <= tol) and np.all(np.abs(b[k]) <= tol) for k in range(a.p)))
