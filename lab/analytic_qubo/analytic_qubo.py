"""Closed-form / local QUBO for the two-intersection traffic model (no 256-schedule cost table).

    offset, Q = build_analytic_qubo(s)            # E(x) = offset + x^T Q x, x = A1..A4 B1..B4, 1 = NS green

Derivation (see DERIVATION.md):

1. Lindley recursion per approach k with green indicators g_t (g = x for NS, 1 - x for EW):
       q_t = max(0, q_{t-1} + a_t + tau_t - c g_t),     q_0 = 0
   D_k(g) = sum_t q_t + w_ov * sum_t (q_t - S)^+.  Only the bits the approach sees enter:
   A approaches -> A1..A4; unlinked B -> B1..B4; linked B -> B1..B4 and A1..A3 through the
   platoon tau_{t+1} = round(f * d^A_t)  (A4's platoon leaves the model).
2. Each D_k is a pseudo-Boolean function of its own <= 7 bits; its multilinear extension
   Dbar_k(p) = E_{x ~ Bernoulli(p)} D_k(x) is a polynomial in p.
3. QUBO = second-order Taylor expansion of every Dbar_k at the fluid (Webster) split p*,
       D_k(x) ~ Dbar_k(p*) + sum_i V_ki (x_i - p*_i) + sum_{i<j} W_kij (x_i - p*_i)(x_j - p*_j)
       V_ki  = E_{p*}[ D_k | x_i = 1 ] - E_{p*}[ D_k | x_i = 0 ]                (expected value of an NS green)
       W_kij = E_{p*}[ D_k(1,1) - D_k(1,0) - D_k(0,1) + D_k(0,0) ]_{ij}           (expected interaction)
   which equals the L2 projection of D_k onto quadratics under the product measure Bernoulli(p*)
   (orthogonality of the centred monomials).  p*_{n,t} = cumulative NS demand share at node n,
       p*_{n,t} = clip( (A^NS_{n,t} + 1/2) / (A^NS_{n,t} + A^EW_{n,t} + 1), 0.25, 0.75 ),
   A^phase_{n,t} = sum_{u<=t} a^phase_{n,u}  (equisaturation / Webster split of the green time);
   the demand of B's linked approach includes the expected platoon f * a^A_{lp, t-1}.
4. Switching w_sw * (x_t XOR x_{t-1}) = w_sw (x_t + x_{t-1} - 2 x_t x_{t-1}) is added exactly.

Work: 16 + 16 + 16 + 128 = 176 single-approach 4-slot recursions (the linked approach depends on
7 bits), never a joint 8-bit schedule.  Variants for ablation: rule='uniform' (p = 1/2, which
reproduces the repo's uniform Walsh projection), rule='red' (naive second-order Moebius
expansion around the all-red state: exact pairwise physics, higher orders dropped),
rule='local_gibbs' (per-node Gibbs measure of each intersection's own local cost).
"""
import itertools
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PHASES = ('NS', 'EW')
COUNTER = {'approach_sims': 0}


def transfer(fraction, departures):
    return int((Decimal(str(fraction)) * Decimal(departures)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def approach(arr, green, c, S, w, extra=(0, 0, 0, 0)):
    """Lindley recursion of one approach: returns (delay + w*overflow, departures)."""
    COUNTER['approach_sims'] += 1
    q, cost, deps = 0, 0.0, []
    for t in range(4):
        avail = q + arr[t] + extra[t]
        d = min(avail, c) if green[t] else 0
        q = avail - d
        cost += q + w * max(0, q - S)
        deps.append(d)
    return cost, deps


def local_components(s):
    """[(name, global variable indices, patterns (2^k, k), exact local cost values)] - local tables only."""
    c, S, w = s['service_capacity'], s['storage_capacity'], s['overflow_weight']
    lp, f = s['link']['phase'], s['link']['fraction']
    up = 'NS' if lp == 'EW' else 'EW'
    arr = s['arrivals']
    green = lambda phase, bits: [int(b) if phase == 'NS' else 1 - int(b) for b in bits]
    P4 = np.array(list(itertools.product([0, 1], repeat=4)))
    P7 = np.array(list(itertools.product([0, 1], repeat=7)))
    comps = []
    for ph in PHASES:
        vals = np.array([approach(arr['A'][ph], green(ph, p), c, S, w)[0] for p in P4])
        comps.append((f'A_{ph}', [0, 1, 2, 3], P4, vals))
    vals = np.array([approach(arr['B'][up], green(up, p), c, S, w)[0] for p in P4])
    comps.append((f'B_{up}', [4, 5, 6, 7], P4, vals))
    # linked B: platoon depends on A1..A3 only -> 8 platoon vectors x 16 B patterns
    tau = {}
    for pa in itertools.product([0, 1], repeat=3):
        _, d = approach(arr['A'][lp], green(lp, list(pa) + [0]), c, S, w)   # A-linked departures (A4 irrelevant)
        COUNTER['approach_sims'] -= 1   # already counted among the 16 A_{lp} recursions
        tau[pa] = [0] + [transfer(f, d[t]) for t in range(3)]
    vals = np.array([approach(arr['B'][lp], green(lp, p[3:]), c, S, w, tau[tuple(p[:3])])[0] for p in P7])
    comps.append((f'B_{lp}_linked', [0, 1, 2, 4, 5, 6, 7], P7, vals))
    return comps


def webster_point(s, clip=0.25, smooth=0.5, link=True):
    """P(NS green) per bit: cumulative demand share (equisaturation split), clipped to [clip, 1-clip].

    Demand of B's linked approach includes the expected platoon f * a^A_{t-1} (link=True).
    """
    p = np.zeros(8)
    lp, f = s['link']['phase'], s['link']['fraction']
    for n, node in enumerate('AB'):
        dem = {ph: np.array(s['arrivals'][node][ph], float) for ph in PHASES}
        if node == 'B' and link:
            dem[lp] = dem[lp] + f * np.r_[0.0, np.array(s['arrivals']['A'][lp][:3], float)]
        ns, ew = np.cumsum(dem['NS']), np.cumsum(dem['EW'])
        p[4 * n:4 * n + 4] = (ns + smooth) / (ns + ew + 2 * smooth)
    return np.clip(p, clip, 1 - clip)


def taylor_coefficients(pats, vals, p):
    """Second-order Taylor expansion of the multilinear extension at p (closed form, exact for product measure).

    Returns (F0, V (k,), W (k,k) upper) with D(x) ~ F0 + sum V_i (x_i-p_i) + sum_{i<j} W_ij (x_i-p_i)(x_j-p_j).
    """
    k = pats.shape[1]
    prob = np.prod(np.where(pats == 1, p, 1 - p), axis=1)
    F0 = float(prob @ vals)
    V = np.zeros(k)
    W = np.zeros((k, k))
    for i in range(k):
        m1, m0 = pats[:, i] == 1, pats[:, i] == 0
        # E[D | x_i=1] - E[D | x_i=0], other bits ~ p
        V[i] = (prob[m1] @ vals[m1]) / p[i] - (prob[m0] @ vals[m0]) / (1 - p[i])
    for i, j in itertools.combinations(range(k), 2):
        e = {}
        for bi, bj in itertools.product([0, 1], repeat=2):
            m = (pats[:, i] == bi) & (pats[:, j] == bj)
            e[bi, bj] = (prob[m] @ vals[m]) / ((p[i] if bi else 1 - p[i]) * (p[j] if bj else 1 - p[j]))
        W[i, j] = e[1, 1] - e[1, 0] - e[0, 1] + e[0, 0]
    return F0, V, W


def _add_taylor(Q, vars_, F0, V, W, p):
    off = F0
    for a, i in enumerate(vars_):
        Q[i, i] += V[a]
        off -= V[a] * p[a]
    for a, b in itertools.combinations(range(len(vars_)), 2):
        i, j = vars_[a], vars_[b]
        v = W[a, b]
        Q[min(i, j), max(i, j)] += v
        Q[i, i] -= v * p[b]
        Q[j, j] -= v * p[a]
        off += v * p[a] * p[b]
    return off


def _add_switching(s, Q):
    w = s['switch_weight']
    for n in (0, 4):
        for t in range(1, 4):
            i, j = n + t - 1, n + t
            Q[i, i] += w
            Q[j, j] += w
            Q[i, j] -= 2 * w


def _wls(pats, vals, wts):
    k = pats.shape[1]
    pairs = list(itertools.combinations(range(k), 2))
    F = np.column_stack([np.ones(len(pats))] + [pats[:, m] for m in range(k)] + [pats[:, a] * pats[:, b] for a, b in pairs])
    sw = np.sqrt(wts)
    coef, *_ = np.linalg.lstsq(F * sw[:, None], vals * sw, rcond=None)
    return coef, pairs


def _mobius_red(name, vars_, pats, vals, s):
    """Naive pairwise physics: Moebius coefficients of order <= 2 around the approach's own all-red state."""
    lp = s['link']['phase']
    k = len(vars_)
    # red value of x for each variable (NS approach red <-> x=0; EW red <-> x=1)
    if name.startswith('A_') or name.startswith('B_') and not name.endswith('linked'):
        ph = name.split('_')[1]
        red = np.full(k, 0 if ph == 'NS' else 1)
    else:
        red = np.full(k, 0 if lp == 'NS' else 1)
    idx = {tuple(r): n for n, r in enumerate(pats)}
    val = lambda flips: vals[idx[tuple(np.where(np.isin(np.arange(k), flips), 1 - red, red))]]
    f0 = val([])
    f1 = np.array([val([i]) - f0 for i in range(k)])
    off = f0
    Qloc = np.zeros((k, k))
    sig = np.where(red == 0, 1.0, -1.0)
    rho = np.where(red == 0, 0.0, 1.0)
    for i in range(k):
        off += f1[i] * rho[i]
        Qloc[i, i] += f1[i] * sig[i]
    for i, j in itertools.combinations(range(k), 2):
        v = val([i, j]) - f1[i] - f1[j] - f0
        off += v * rho[i] * rho[j]
        Qloc[i, i] += v * sig[i] * rho[j]
        Qloc[j, j] += v * rho[i] * sig[j]
        Qloc[i, j] += v * sig[i] * sig[j]
    return off, Qloc


def build_analytic_qubo(s, rule='webster', clip=0.25, link=True, kappa=1.0, return_info=False):
    COUNTER['approach_sims'] = 0
    comps = local_components(s)
    Q = np.zeros((8, 8))
    off = 0.0
    if rule in ('webster', 'uniform'):
        p8 = webster_point(s, clip=clip, link=link) if rule == 'webster' else np.full(8, 0.5)
        for name, vars_, pats, vals in comps:
            p = p8[vars_]
            F0, V, W = taylor_coefficients(pats, vals, p)
            off += _add_taylor(Q, vars_, F0, V, W, p)
    elif rule == 'red':
        for name, vars_, pats, vals in comps:
            o, Ql = _mobius_red(name, vars_, pats, vals, s)
            off += o
            for a, i in enumerate(vars_):
                for b, j in enumerate(vars_):
                    if Ql[a, b]:
                        Q[min(i, j), max(i, j)] += Ql[a, b]
    elif rule == 'local_gibbs':
        # per-node Gibbs measure of each intersection's own local cost (B's linked cost averaged over A uniformly)
        P4 = comps[0][2]
        swc = s['switch_weight'] * np.array([sum(p[t] != p[t - 1] for t in range(1, 4)) for p in P4])
        cA = comps[0][3] + comps[1][3] + swc
        pats7, vals7 = comps[3][2], comps[3][3]
        bidx = pats7[:, 3:] @ [8, 4, 2, 1]
        cB = comps[2][3] + np.array([vals7[bidx == m].mean() for m in range(16)]) + swc
        wA = np.exp(-(cA - cA.min()) / (kappa * cA.std()))
        wB = np.exp(-(cB - cB.min()) / (kappa * cB.std()))
        for name, vars_, pats, vals in comps:
            if name.startswith('A_'):
                w = wA[pats @ [8, 4, 2, 1]]
            elif name.endswith('linked'):
                w = wA.reshape(8, 2).sum(1)[pats[:, :3] @ [4, 2, 1]] * wB[pats[:, 3:] @ [8, 4, 2, 1]]
            else:
                w = wB[pats @ [8, 4, 2, 1]]
            coef, pairs = _wls(pats, vals, w)
            k = len(vars_)
            off += coef[0]
            for m, i in enumerate(vars_):
                Q[i, i] += coef[1 + m]
            for n_, (a, b) in enumerate(pairs):
                i, j = vars_[a], vars_[b]
                Q[min(i, j), max(i, j)] += coef[1 + k + n_]
    else:
        raise ValueError(rule)
    _add_switching(s, Q)
    Q[np.abs(Q) < 1e-9] = 0.0
    if return_info:
        return float(off), Q, {'approach_sims': COUNTER['approach_sims'], 'rule': rule}
    return float(off), Q
