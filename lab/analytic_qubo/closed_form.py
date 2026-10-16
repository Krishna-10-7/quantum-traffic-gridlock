"""Closed-form low-order physics of one approach (anchored at the all-red state) + verification.

For an approach with arrivals a_t (cumulative A_t = sum_{u<=t} a_u), capacity c, storage S and
overflow weight w, with only the greens in G switched on, queues are explicit:

  no green          q_t = A_t
  green at s        q_t = A_t - delta_s [t>=s],                delta_s   = min(c, A_s)
  greens at s < t   q_u = A_u - delta_s [u>=s] - delta_t|s [u>=t],  delta_t|s = min(c, A_t - delta_s)

and Phi(q) = sum_t q_t + w (q_t - S)^+.  Moebius coefficients (order <= 2) are finite differences of
Phi on these vectors.  Delay parts in closed form:

  value of one green          alpha_s   = -(5 - s) delta_s
  interference of two greens  beta_st   = (5 - t) (min(c, A_t) - min(c, A_t - delta_s)) >= 0
  A->B platoon of an A green  P_s       = round(f * min(c, A^A_s))  enters B at slot s+1 (s <= 3)
     B all red:               +(4 - s) P_s vehicle-slots at B
     B green at t > s:        -(5 - t) (min(c, A^B_t + P_s) - min(c, A^B_t))   ("green wave" coupling)
     two A greens s < s':     (4 - s') (round(f min(c, A^A_s' - delta_s)) - P_s')  <= 0

This file checks these formulas against brute-force finite differences of the Lindley recursion.
    python lab/analytic_qubo/closed_form.py
"""
import itertools
import os
import sys
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import traffic_model as tm  # noqa: E402
from analytic_qubo import approach, transfer  # noqa: E402


def delay_only(arr, green, c, extra=(0, 0, 0, 0)):
    return approach(arr, green, c, 10 ** 9, 0.0, extra)


def check(n=300, seed=7):
    scs = tm.random_scenarios(n, seed=seed, min_gap=0.0)
    worst = 0.0
    for s in scs:
        c, f, lp = s['service_capacity'], s['link']['fraction'], s['link']['phase']
        for node, ph in itertools.product('AB', ('NS', 'EW')):
            a = s['arrivals'][node][ph]
            A = [int(v) for v in np.cumsum(a)]
            F = lambda G: delay_only(a, [int(t in G) for t in range(4)], c)[0]
            f0 = F(())
            for s_ in range(4):
                alpha = F((s_,)) - f0
                worst = max(worst, abs(alpha - (-(4 - s_) * min(c, A[s_]))))      # slots are 0-based here: 5-s -> 4-s_
                for t in range(s_ + 1, 4):
                    beta = F((s_, t)) - F((s_,)) - F((t,)) + f0
                    ds = min(c, A[s_])
                    pred = (4 - t) * (min(c, A[t]) - min(c, A[t] - ds))
                    worst = max(worst, abs(beta - pred))
        # link terms
        aA, aB = s['arrivals']['A'][lp], s['arrivals']['B'][lp]
        AA, AB = [int(v) for v in np.cumsum(aA)], [int(v) for v in np.cumsum(aB)]

        def FL(GA, GB):
            _, d = delay_only(aA, [int(t in GA) for t in range(4)], c)
            tau = [0] + [transfer(f, d[t]) for t in range(3)]
            return delay_only(aB, [int(t in GB) for t in range(4)], c, tau)[0]
        f0 = FL((), ())
        for s_ in range(3):
            P = transfer(f, min(c, AA[s_]))
            worst = max(worst, abs((FL((s_,), ()) - f0) - (3 - s_) * P))
            for t in range(4):
                inter = FL((s_,), (t,)) - FL((s_,), ()) - FL((), (t,)) + f0
                pred = 0 if t <= s_ else -(4 - t) * (min(c, AB[t] + P) - min(c, AB[t]))
                worst = max(worst, abs(inter - pred))
            for s2 in range(s_ + 1, 3):
                inter = FL((s_, s2), ()) - FL((s_,), ()) - FL((s2,), ()) + f0
                ds = min(c, AA[s_])
                pred = (3 - s2) * (transfer(f, min(c, AA[s2] - ds)) - transfer(f, min(c, AA[s2])))
                worst = max(worst, abs(inter - pred))
    return worst


if __name__ == '__main__':
    w = check()
    print(f'max |closed form - brute force finite difference| over 300 scenarios: {w:.2e}')
    assert w < 1e-9
