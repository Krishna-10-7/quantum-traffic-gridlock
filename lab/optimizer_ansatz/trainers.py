"""Budgeted QAOA trainers explored in this lab.

Every trainer goes through `Budget`, which counts EVERY statevector evaluation of the training
objective (probes, initial points, line-search points) and raises once max_evals is reached.
A call with exactly the same parameter vector as the last evaluated point of the same stage is
served from a one-entry memo and is not a new evaluation (no circuit is run).

Objectives are functions of the QUBO energies only (qaoa.make_objective); the official cost is
never touched here.

Stage-2 optimiser 'nft' = sequential minimal optimisation (Nakanishi, Fujii & Todo 2020,
"Rotosolve", Ostaszewski et al. 2021) exploiting the fact that the Gibbs objective is
f = -log <psi(theta)| B |psi(theta)>,  B = diag(exp(-eta (E - Emin)/std E)),
and the expectation <B> is an exact sinusoid  c + a cos(w t) + b sin(w t)  in every parameter t
that enters the circuit as exp(-i t (w/2) P) with P a Pauli string:
  * per-qubit mixer angle   b_{k,i}: exp(+i b X_i)              -> w = 2
  * per-field cost angle    g_{k,i}: exp(-i g h_i Z_i / sigma)  -> w = 2 |h_i| / sigma   (cost_mode='field')
Two new evaluations (t +- pi/(2w)) plus the current value determine the 1-D function exactly,
and the coordinate is moved to its exact 1-D global optimum. The remaining (generic) angles
(the shared field / coupling angles) are refined by short COBYLA blocks restricted to them.
"""
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", message=".*MAXFUN.*")

import qaoa
import qubo as qb

N = qb.N


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_evals):
        self.max, self.n, self.hist = max_evals, 0, []
        self.best_v, self.best_x = np.inf, None   # best in FINAL-ansatz coordinates
        self.best_raw = None                      # same point in the coordinates of the stage that produced it

    def wrap(self, f, embed=None):
        memo = {'x': None, 'v': None}

        def g(x):
            x = np.asarray(x, float)
            if memo['x'] is not None and np.array_equal(x, memo['x']):
                return memo['v']
            if self.n >= self.max:
                raise BudgetExceeded
            v = float(f(x))
            self.n += 1
            self.hist.append(v)
            memo.update(x=x.copy(), v=v)
            if v < self.best_v:
                self.best_v = v
                self.best_x = (embed(x) if embed else x.copy())
                self.best_raw = x.copy()
            return v
        return g

    @property
    def left(self):
        return self.max - self.n


def _run(fun, x0, method, evals, rho, tol=1e-7):
    if evals <= 0:
        return
    try:
        if method == 'cobyla':
            minimize(fun, x0, method='COBYLA', options={'maxiter': evals, 'rhobeg': rho, 'tol': tol})
        elif method == 'cobyqa':
            minimize(fun, x0, method='COBYQA', options={'maxfev': evals, 'initial_tr_radius': rho,
                                                       'final_tr_radius': 1e-8})
        elif method == 'powell':
            minimize(fun, x0, method='Powell', options={'maxfev': evals, 'xtol': 1e-6, 'ftol': 1e-10,
                                                       'direc': np.eye(len(x0)) * rho})
        elif method == 'nelder-mead':
            sim = np.vstack([x0] + [x0 + rho * e for e in np.eye(len(x0))])
            minimize(fun, x0, method='Nelder-Mead', options={'maxfev': evals, 'initial_simplex': sim,
                                                            'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True})
        else:
            raise ValueError(method)
    except BudgetExceeded:
        pass


# ------------------------------------------------------------------ ansatz helpers
def param_structure(a):
    """(sinusoidal index list, frequency list, generic index list) for an Ansatz."""
    p, mg, mb = a.p, a.mg, a.mb
    sin_idx, freq, gen = [], [], []
    for k in range(p):
        for j in range(mg):
            idx = k * mg + j
            if a.cost_mode == 'field' and j < N:
                w = 2 * abs(a.h[j]) / a.scale
                if w > 1e-12:
                    sin_idx.append(idx)
                    freq.append(w)
            else:
                gen.append(idx)
        for i in range(mb):
            idx = p * mg + k * mb + i
            if a.mixer_mode == 'qubit':
                sin_idx.append(idx)
                freq.append(2.0)
            else:
                gen.append(idx)
    return sin_idx, freq, gen


def interp(params, p):
    """INTERP (Zhou et al. 2020): standard-QAOA angles at depth p -> depth p+1."""
    g, b = params[:p], params[p:]

    def up(v):
        v = np.concatenate([[0.0], v, [0.0]])
        return np.array([(i - 1) / p * v[i - 1] + (p - i + 1) / p * v[i] for i in range(1, p + 2)])
    return np.concatenate([up(g), up(b)])


def embed_base(final, base_p, x, p_final):
    """Standard-QAOA params at depth base_p (<= p_final) -> final ansatz params (zero-padded layers would
    change the state, so only base_p == p_final is embedded exactly)."""
    assert base_p == p_final
    base = qaoa.Ansatz(final.h, final.J, p_final, 'single', 'single')
    return final.lift_params(x, base)


# ------------------------------------------------------------------ NFT sweep
def nft_sweep(f, x, fx, idx, freq, budget, kind, order='forward', max_n=None):
    """One sequential-minimal-optimisation sweep over sinusoidal coordinates.
    f counts evaluations. Returns (x, fx). fx is the exact (analytic) value at the new x."""
    to_y = (lambda v: np.exp(-v)) if kind == 'gibbs' else (lambda v: -v)
    from_y = (lambda y: -np.log(y)) if kind == 'gibbs' else (lambda y: -y)
    seq = list(zip(idx, freq))
    if order == 'backward':
        seq = seq[::-1]
    for j, w in seq:
        if budget.left < 2 or (max_n is not None and budget.n + 2 > max_n):
            break
        d = (np.pi / 2) / w
        xp, xm = x.copy(), x.copy()
        xp[j] += d
        xm[j] -= d
        y0, yp, ym = to_y(fx), to_y(f(xp)), to_y(f(xm))
        c = 0.5 * (yp + ym)
        A, B = y0 - c, 0.5 * (yp - ym)
        R = np.hypot(A, B)
        if R < 1e-15:
            continue
        t = np.arctan2(B, A) / w
        x = x.copy()
        x[j] += t
        fx = float(from_y(c + R))
    return x, fx


# ------------------------------------------------------------------ main trainer
DEFAULT = dict(p=4, cost_mode='split', mixer_mode='qubit', kind='gibbs', eta=2.0, alpha=0.25, max_evals=300,
               ramp_dts=(0.75, 1.0, 1.25), stage1='cobyla', stage1_evals=40, rhobeg=0.15,
               stage2='cobyla', rho2=None, layerwise=None,
               nft_sweeps=None, gen_evals=16, gen_rho=0.05, nft_order='forward', final_polish=0,
               eta_stage1=None, restarts=1, mid=None, mid_evals=0, mid_opt='cobyla', gen_first=False)


def train(h, J, E, cfg=None):
    c = {**DEFAULT, **(cfg or {})}
    p, kind = c['p'], c['kind']
    budget = Budget(c['max_evals'])
    final = qaoa.Ansatz(h, J, p, c['cost_mode'], c['mixer_mode'])
    eta1 = c['eta_stage1'] if c['eta_stage1'] is not None else c['eta']
    rho2 = c['rho2'] if c['rho2'] is not None else c['rhobeg'] / 2

    # ---------------- stage 1: standard QAOA (2p params)
    if c['layerwise']:
        # layer-wise growth p=1..p with INTERP; c['layerwise'] = cumulative eval budget after each depth
        x = None
        for q in range(1, p + 1):
            a = qaoa.Ansatz(h, J, q, 'single', 'single')
            emb = (lambda xx, a=a: final.lift_params(xx, a)) if q == p else (lambda xx: None)
            fq = budget.wrap(qaoa.make_objective(a, E, kind, c['alpha'], eta1), emb if q == p else None)
            if q == 1:
                starts = [a.ramp(dt) for dt in c['ramp_dts']]
                vals = [fq(s0) for s0 in starts]
                x = starts[int(np.argmin(vals))]
            else:
                x = interp(x, q - 1)
            # track best within this depth only
            loc = {'v': np.inf, 'x': x}

            def fl(xx, fq=fq, loc=loc):
                v = fq(xx)
                if v < loc['v']:
                    loc.update(v=v, x=np.array(xx, float))
                return v
            _run(fl, x, 'cobyla', c['layerwise'][q - 1] - budget.n, c['rhobeg'])
            if loc['v'] == np.inf:
                fl(x)
            x = loc['x']
            if q < p:
                budget.best_v, budget.best_x = np.inf, None   # earlier depths are not final-ansatz states
        x_base = x
    else:
        base = qaoa.Ansatz(h, J, p, 'single', 'single')
        fb = budget.wrap(qaoa.make_objective(base, E, kind, c['alpha'], eta1), lambda xx: final.lift_params(xx, base))
        starts = [base.ramp(dt) for dt in c['ramp_dts']]
        vals = [fb(s0) for s0 in starts]
        order = np.argsort(vals)
        if c['restarts'] > 1:
            # split stage-1 budget over the best `restarts` probes
            per = (c['stage1_evals'] - budget.n) // c['restarts']
            for r in range(c['restarts']):
                loc = {'v': np.inf, 'x': None}

                def fl(xx, loc=loc):
                    v = fb(xx)
                    if v < loc['v']:
                        loc.update(v=v, x=np.array(xx, float))
                    return v
                _run(fl, starts[order[r % len(starts)]], c['stage1'], per, c['rhobeg'])
        else:
            _run(fb, starts[order[0]], c['stage1'], c['stage1_evals'] - budget.n, c['rhobeg'])
        x_base = None

    rich = (c['cost_mode'], c['mixer_mode']) != ('single', 'single')
    ff = budget.wrap(qaoa.make_objective(final, E, kind, c['alpha'], c['eta']))
    x = budget.best_x.copy()
    if not rich and c['stage2'] == 'cobyla':
        _run(ff, x, 'cobyla', budget.left, c['rhobeg'])
        return final, budget.best_x, budget

    # ---------------- optional middle stage, e.g. mid=('single','qubit'): per-qubit mixer, shared gamma
    if c['mid']:
        base = qaoa.Ansatz(h, J, p, 'single', 'single')
        mid = qaoa.Ansatz(h, J, p, c['mid'][0], c['mid'][1])
        xm = mid.lift_params(budget.best_raw, base)
        fm = budget.wrap(qaoa.make_objective(mid, E, kind, c['alpha'], eta1), lambda xx: final.lift_params(xx, mid))
        fm(xm)
        if c['mid_opt'] == 'nft':
            sidx, freq, _ = param_structure(mid)
            fx = budget.best_v
            stop = budget.n + c['mid_evals']
            while budget.left >= 2 and budget.n + 2 <= stop:
                xm, fx = nft_sweep(fm, xm, fx, sidx, freq, budget, kind, c['nft_order'], max_n=stop)
                budget.best_v, budget.best_x = fx, final.lift_params(xm, mid)
        else:
            _run(fm, xm, 'cobyla', c['mid_evals'] - 1, rho2)
        x = budget.best_x.copy()

    # ---------------- stage 2
    budget.best_v = np.inf   # objective may change (eta schedule): restart best tracking in the final objective
    fx = ff(x)
    if c['stage2'] in ('cobyla', 'cobyqa', 'powell', 'nelder-mead'):
        _run(ff, x, c['stage2'], budget.left, rho2)
    elif c['stage2'] == 'nft':
        sidx, freq, gen = param_structure(final)
        sweeps = 0
        while budget.left >= 2:
            if gen and c['gen_evals'] and (sweeps > 0 or c['gen_first']):
                xg = x.copy()

                def fg(z, xg=xg):
                    xx = xg.copy()
                    xx[gen] = z
                    return ff(xx)
                n_left = budget.left
                _run(fg, x[gen], 'cobyla', min(c['gen_evals'], budget.left), c['gen_rho'])
                x, fx = budget.best_x.copy(), budget.best_v
            x, fx = nft_sweep(ff, x, fx, sidx, freq, budget, kind, c['nft_order'])
            budget.best_v, budget.best_x = fx, x.copy()
            sweeps += 1
            if c['nft_sweeps'] and sweeps >= c['nft_sweeps']:
                break
            if c['final_polish'] and budget.left <= c['final_polish']:
                break
        if budget.left > 0 and c['final_polish']:
            _run(ff, x, 'cobyla', budget.left, c['gen_rho'])
    else:
        raise ValueError(c['stage2'])
    return final, budget.best_x, budget


def _nft_loop(ff, x, fx, a, budget, kind, c, stop):
    """NFT sweeps on the sinusoidal coordinates of ansatz a, short COBYLA blocks on its generic ones,
    until budget.n reaches `stop` (or the global budget)."""
    sidx, freq, gen = param_structure(a)
    sweeps = 0
    while budget.left >= 2 and budget.n + 2 <= stop:
        if gen and c['gen_evals'] and (sweeps > 0 or c['gen_first']):
            xg = x.copy()

            def fg(z, xg=xg):
                xx = xg.copy()
                xx[gen] = z
                return ff(xx)
            _run(fg, x[gen], 'cobyla', min(c['gen_evals'], budget.left, stop - budget.n), c['gen_rho'])
            x, fx = budget.best_x.copy(), budget.best_v
        x, fx = nft_sweep(ff, x, fx, sidx, freq, budget, kind, c['nft_order'], max_n=stop)
        budget.best_v, budget.best_x = fx, x.copy()
        sweeps += 1
    return x, fx


def train_grow(h, J, E, cfg=None):
    """Layer-wise growth for the multi-angle ansatz: depth 1 -> p_final.

    cfg['grow'] = cumulative evaluation counts at which one identity-initialised layer
    (all its angles = 0, so the state and objective are unchanged) is appended; the NFT/COBYLA
    loop then continues on all parameters of the deeper ansatz. Stage 1 is standard QAOA at p=1.
    Returns the final-depth ansatz and parameters."""
    c = {**DEFAULT, **(cfg or {})}
    kind = c['kind']
    budget = Budget(c['max_evals'])
    thresholds = list(c['grow']) + [c['max_evals']]
    p = 1
    a = qaoa.Ansatz(h, J, 1, c['cost_mode'], c['mixer_mode'])
    base = qaoa.Ansatz(h, J, 1, 'single', 'single')
    fb = budget.wrap(qaoa.make_objective(base, E, kind, c['alpha'], c['eta']), lambda xx: a.lift_params(xx, base))
    starts = [base.ramp(dt) for dt in c['ramp_dts']]
    vals = [fb(s0) for s0 in starts]
    _run(fb, starts[int(np.argmin(vals))], 'cobyla', c['stage1_evals'] - budget.n, c['rhobeg'])
    x, fx = budget.best_x.copy(), budget.best_v
    budget.best_v = np.inf
    ff = budget.wrap(qaoa.make_objective(a, E, kind, c['alpha'], c['eta']))
    fx = ff(x)
    for stop in thresholds:
        x, fx = _nft_loop(ff, x, fx, a, budget, kind, c, stop)
        if p == c['p'] or stop >= c['max_evals']:
            break
        # append an identity layer
        g, b = a.split(x)
        a = qaoa.Ansatz(h, J, p + 1, c['cost_mode'], c['mixer_mode'])
        p += 1
        x = a.join(np.vstack([g, np.zeros((1, g.shape[1]))]), np.vstack([b, np.zeros((1, b.shape[1]))]))
        ff = budget.wrap(qaoa.make_objective(a, E, kind, c['alpha'], c['eta']))
        budget.best_v, budget.best_x = fx, x.copy()   # same state, exact same objective value
    while p < c['p']:   # budget exhausted before reaching full depth: pad with identity layers
        g, b = a.split(x)
        a = qaoa.Ansatz(h, J, p + 1, c['cost_mode'], c['mixer_mode'])
        p += 1
        x = a.join(np.vstack([g, np.zeros((1, g.shape[1]))]), np.vstack([b, np.zeros((1, b.shape[1]))]))
        budget.best_x = x.copy()
    return a, budget.best_x, budget
