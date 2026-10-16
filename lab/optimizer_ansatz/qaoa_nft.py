"""Drop-in trainer for qaoa.py: budgeted sequential minimal optimisation (NFT / Rotosolve)
for the multi-angle QAOA ansatz. Same signature and return value as qaoa.train, so

    ansatz, params, history = qaoa_nft.train(h, J, energies, p=4, cost_mode='field', mixer_mode='qubit', ...)

replaces qaoa.train(...) in run.solve without any other change.

Why it works. In qaoa.Ansatz every per-qubit mixer angle b_{k,i} enters as exp(+i b X_i), and
(cost_mode='field') every per-field cost angle g_{k,i} enters as exp(-i g (h_i/sigma) Z_i). For such a
parameter t, with all others fixed, the state is psi(t) = W2 exp(-i t (w/2) P) W1 |+>^8 with P a
Pauli operator (P^2 = I), so for ANY diagonal operator B

    <B>(t) = c + a cos(w t) + b sin(w t),     w = 2 (mixer),  w = 2|h_i|/sigma (field).

The Gibbs objective is f = -log<B>, B = diag(exp(-eta (E - Emin)/std E)) (QUBO energies only), a
strictly decreasing function of <B>; the energy objective is f = <E>. Evaluating f at t0 + pi/(2w)
and t0 - pi/(2w) (2 evaluations; f(t0) is known) fixes (a, b, c) exactly, and the coordinate is set
to the exact 1-D global optimum t* = t0 + atan2(b', a')/w, whose value c + sqrt(a'^2 + b'^2) is known
analytically (Nakanishi, Fujii & Todo, PRR 2, 043158 (2020); Ostaszewski et al., Quantum 5, 391 (2021)).
The few remaining "generic" angles (the shared coupling angle per layer, and the shared field angle in
cost_mode='split') are not single-frequency; they are refined by short COBYLA blocks restricted to them.

Schedule (default = the frozen lab candidate "field-nft p=4 direct, stage1 10, gen16"):
  stage 1  3 linear-ramp probes + COBYLA on standard QAOA (2p angles) until `stage1_evals` evaluations;
  stage 2  embed the best point into the multi-angle ansatz (identical state), then repeat
           [COBYLA block of `gen_evals` evaluations on the generic angles (skipped before the 1st sweep)]
           [one NFT sweep, layer by layer, over all sinusoidal angles (2 evaluations each)]
           until the hard cap `max_evals`.

Budget: every statevector evaluation is counted and appended to `history` (hard cap max_evals); the
probes and the re-evaluation of the embedded point count. The analytic value at t* is not an evaluation.
A call that repeats the last evaluated parameter vector of the same stage is served from a one-entry memo
(no circuit is run, nothing is counted).

Fallbacks (keep the repository's ablations runnable): objectives that are not a monotone function of one
expectation value (kind='cvar') or ansatzes without sinusoidal coordinates (cost_mode='single' with
mixer_mode='single') are delegated unchanged to qaoa.train.
"""
import numpy as np
from scipy.optimize import minimize

import qaoa


class BudgetExceeded(Exception):
    pass


def sinusoidal_coordinates(a):
    """Indices of parameters that enter as exp(-i t (w/2) P) with a single Pauli P, their w,
    and the remaining (generic) indices, for a qaoa.Ansatz. Ordered layer by layer."""
    p, mg, mb, n = a.p, a.mg, a.mb, qaoa.N
    sin_idx, freq, gen = [], [], []
    for k in range(p):
        for j in range(mg):
            idx = k * mg + j
            if a.cost_mode == 'field' and j < n:
                w = 2 * abs(a.h[j]) / a.scale
                if w > 1e-12:          # h_j == 0: the angle does not act at all
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


def _starts(base, init, init_seed, ramp_dts):
    """Initial points, identical to qaoa.train."""
    if init == 'ramp':
        return [base.ramp(dt) for dt in ramp_dts]
    if init == 'random':
        rng = np.random.default_rng(init_seed)
        return [rng.uniform(0, np.pi / 2, base.n_params) for _ in ramp_dts]
    if init == 'small':
        return [np.full(base.n_params, 0.05)]
    raise ValueError(init)


def train(h, J, energies, p=4, cost_mode='field', mixer_mode='qubit', kind='gibbs', eta=2.0, alpha=0.25,
          max_evals=300, ramp_dts=(0.75, 1.0, 1.25), stage1_evals=10, rhobeg=0.15, init='ramp', init_seed=0,
          gen_evals=16, gen_rho=0.05):
    """Returns (ansatz, best_params, history); len(history) <= max_evals counts every evaluation."""
    final = qaoa.Ansatz(h, J, p, cost_mode, mixer_mode)
    sin_idx, freq, gen = sinusoidal_coordinates(final)
    if kind not in ('gibbs', 'energy') or not sin_idx:
        return qaoa.train(h, J, energies, p=p, cost_mode=cost_mode, mixer_mode=mixer_mode, kind=kind, eta=eta,
                          alpha=alpha, max_evals=max_evals, ramp_dts=ramp_dts, stage1_evals=stage1_evals,
                          rhobeg=rhobeg, init=init, init_seed=init_seed)
    base = qaoa.Ansatz(h, J, p, 'single', 'single')
    history = []
    best = {'v': np.inf, 'x': None}

    def counted(f, embed=None):
        memo = {'x': None, 'v': None}

        def g(x):
            x = np.asarray(x, float)
            if memo['x'] is not None and np.array_equal(x, memo['x']):
                return memo['v']
            if len(history) >= max_evals:
                raise BudgetExceeded
            v = float(f(x))
            history.append(v)
            memo.update(x=x.copy(), v=v)
            if v < best['v']:
                best.update(v=v, x=embed(x) if embed else x.copy())
            return v
        return g

    def cobyla(f, x0, evals, rho):
        if evals <= 0:
            return
        try:
            minimize(f, x0, method='COBYLA', options={'maxiter': evals, 'rhobeg': rho, 'tol': 1e-7})
        except BudgetExceeded:
            pass

    # ---- stage 1: standard QAOA from the best initial probe
    fb = counted(qaoa.make_objective(base, energies, kind, alpha, eta), lambda xx: final.lift_params(xx, base))
    starts = _starts(base, init, init_seed, ramp_dts)
    vals = [fb(s0) for s0 in starts]
    cobyla(fb, starts[int(np.argsort(vals)[0])], stage1_evals - len(history), rhobeg)

    # ---- stage 2: multi-angle ansatz, NFT sweeps + COBYLA blocks on generic angles
    ff = counted(qaoa.make_objective(final, energies, kind, alpha, eta))
    x = best['x'].copy()
    best['v'] = np.inf
    fx = ff(x)
    to_y = (lambda v: np.exp(-v)) if kind == 'gibbs' else (lambda v: -v)
    from_y = (lambda y: -np.log(y)) if kind == 'gibbs' else (lambda y: -y)
    sweeps = 0
    while max_evals - len(history) >= 2:
        n_before = len(history)
        if gen and gen_evals and sweeps > 0:
            xg = x.copy()

            def fg(z, xg=xg):
                xx = xg.copy()
                xx[gen] = z
                return ff(xx)
            cobyla(fg, x[gen], min(gen_evals, max_evals - len(history)), gen_rho)
            x, fx = best['x'].copy(), best['v']
        for j, w in zip(sin_idx, freq):
            if max_evals - len(history) < 2:
                break
            d = (np.pi / 2) / w
            xp, xm = x.copy(), x.copy()
            xp[j] += d
            xm[j] -= d
            y0, yp, ym = to_y(fx), to_y(ff(xp)), to_y(ff(xm))
            c = 0.5 * (yp + ym)
            A, B = y0 - c, 0.5 * (yp - ym)
            R = np.hypot(A, B)
            if R < 1e-15:
                continue
            x = x.copy()
            x[j] += np.arctan2(B, A) / w
            fx = float(from_y(c + R))
        best.update(v=fx, x=x.copy())
        sweeps += 1
        if len(history) == n_before:   # nothing left that consumes evaluations (degenerate ansatz)
            break
    assert len(history) <= max_evals
    return final, best['x'], history


train_nft = train   # backwards-compatible name used by earlier lab scripts
