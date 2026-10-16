"""Traffic model decomposed by approach, plus scenario utilities.

The official cost is a sum of per-approach terms (delay + overflow) and a
per-intersection switching term. Each approach's term only depends on the
green bits it can "see":

  * A-NS, A-EW               -> the four A bits
  * the unlinked B approach  -> the four B bits
  * the linked B approach    -> the four B bits AND the four A bits (via transfers)

This locality is what fixes the sparsity pattern of the QUBO (see qubo.py).
Bit order everywhere: x[0..3] = A slots 1-4, x[4..7] = B slots 1-4, 1 = NS green.
"""
from decimal import Decimal, ROUND_HALF_UP
import itertools
import json
import random

import numpy as np

import scorer

NODES = ('A', 'B')
PHASES = ('NS', 'EW')
ALL_BITS = np.array(list(itertools.product([0, 1], repeat=8)), dtype=int)  # row i = bits of int i, MSB first
BITSTRINGS = [''.join(map(str, b)) for b in ALL_BITS]


def load_scenario(path):
    with open(path) as fh:
        s = json.load(fh)
    scorer.validate_scenario(s)
    return s


def transfer(fraction, departures):
    """floor(fraction * departures + 0.5) with exact decimal half-up, as in scorer.py."""
    amt = Decimal(str(fraction)) * Decimal(departures)
    return int(amt.quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def approach_trace(s, node, phase, plan, inflow_extra=(0, 0, 0, 0)):
    """Simulate one approach. Returns (delay, overflow, departures per slot)."""
    cap, store = s['service_capacity'], s['storage_capacity']
    q = delay = overflow = 0
    deps = []
    for t in range(4):
        avail = q + s['arrivals'][node][phase][t] + inflow_extra[t]
        green = plan[t] == (1 if phase == 'NS' else 0)
        d = min(avail, cap) if green else 0
        q = avail - d
        delay += q
        overflow += max(0, q - store)
        deps.append(d)
    return delay, overflow, deps


def approach_costs(s, bits):
    """Per-approach weighted cost terms and switching cost for one 8-bit schedule.

    Returns dict {('A','NS'): c, ..., 'switch': c}. Their sum equals scorer.evaluate cost.
    """
    a, b = list(bits[:4]), list(bits[4:])
    w = s['overflow_weight']
    link_phase, frac = s['link']['phase'], s['link']['fraction']
    out = {}
    pending = [0, 0, 0, 0]
    for phase in PHASES:
        dl, ov, deps = approach_trace(s, 'A', phase, a)
        out[('A', phase)] = dl + w * ov
        if phase == link_phase:
            for t in range(3):
                pending[t + 1] = transfer(frac, deps[t])
    for phase in PHASES:
        extra = pending if phase == link_phase else (0, 0, 0, 0)
        dl, ov, _ = approach_trace(s, 'B', phase, b, extra)
        out[('B', phase)] = dl + w * ov
    sw = sum(plan[t] != plan[t - 1] for plan in (a, b) for t in range(1, 4))
    out['switch'] = s['switch_weight'] * sw
    return out


def cost(s, bits):
    return float(sum(approach_costs(s, bits).values()))


def cost_table(s):
    """Official cost of all 256 schedules, indexed by int(bitstring, 2) (A1 is the MSB)."""
    return np.array([scorer.evaluate(s, [int(v) for v in row])['cost'] for row in ALL_BITS])


def random_scenario(rng, name='rand'):
    """Draw a scenario from the hidden-scenario ranges stated in the problem statement."""
    return {
        'name': name,
        'arrivals': {n: {p: [rng.randint(0, 8) for _ in range(4)] for p in PHASES} for n in NODES},
        'service_capacity': rng.randint(4, 6),
        'storage_capacity': rng.randint(5, 8),
        'overflow_weight': rng.choice([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]),
        'switch_weight': rng.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
        'link': {'from': 'A', 'to': 'B', 'phase': rng.choice(PHASES),
                 'fraction': rng.choice([0.4, 0.5, 0.6, 0.7, 0.8])},
    }


def random_scenarios(n, seed=0, min_gap=5.0):
    """n random scenarios whose C_rand - C* exceeds min_gap (organizers drop tiny denominators)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = random_scenario(rng, name=f'rand{len(out)}')
        c = cost_table(s)
        if c.mean() - c.min() > min_gap:
            out.append(s)
    return out
