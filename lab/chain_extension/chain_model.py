"""Vectorised chain dynamics, per-approach decomposition, light-cone scopes and scenario generator.

Bit order: x[0..3] = A1..A4, x[4..7] = B1..B4, x[8..11] = C1..C4 (1 = NS green).
Row i of ALL_BITS is the binary expansion of i with A1 as the most significant bit, so a
schedule string's integer value is its row index (same convention as traffic_model.py).

Decomposition (exact): C(x) = sum_k D_k(x_{S_k}) + w_sw * sum_n sum_t XOR(x_{n,t}, x_{n,t-1}),
k over the six approaches (A,B,C) x (NS,EW), D_k = delay_k + w_ov * overflow_k.

Light cone. A transfer leaves node n in slot t and enters n+1 in slot t+1, and transfers after
the last slot leave the model, so the approach of node n reached through a chain of m aligned links
can only depend on the upstream node n-m's bits in slots 1..4-m. Hence
   S(A,*)               = {A1..A4}
   S(B, unlinked)       = {B1..B4}
   S(B, phase of A->B)  = {B1..B4, A1..A3}
   S(C, unlinked)       = {C1..C4}
   S(C, phase of B->C)  = {C1..C4, B1..B3}                 if B->C phase != A->B phase
                        = {C1..C4, B1..B3, A1..A2}         if the two links use the same phase (corridor)
For a corridor of any length with T slots, |S_k| <= T + (T-1) + ... + 1 = T(T+1)/2 = 10.
"""
import itertools
import random

import numpy as np

import chain_scorer as cs

NODES, PHASES = cs.NODES, cs.PHASES
N = 12
NODE_OFF = {'A': 0, 'B': 4, 'C': 8}
ALL_BITS = np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.int8)
BITSTRINGS = [''.join(map(str, b)) for b in ALL_BITS]


def _transfer_lut(fraction, cap):
    return np.array([cs.transfer(fraction, d) for d in range(cap + 1)], dtype=np.int64)


def approach_tables(s, bits=ALL_BITS):
    """Exact per-approach cost terms for every row of `bits` (vectorised scorer recurrence).

    Returns {(node, phase): D_k array, 'switch': w_sw * switches array}; the sum is the official chain cost.
    """
    cs.validate_chain(s)
    bits = np.asarray(bits, dtype=np.int64)
    M = bits.shape[0]
    cap, store, w = s['service_capacity'], s['storage_capacity'], s['overflow_weight']
    link_in = {l['to']: l for l in s['links']}
    link_out = {l['from']: l for l in s['links']}
    lut = {l['from']: _transfer_lut(l['fraction'], cap) for l in s['links']}
    pending = {'B': np.zeros((M, 4), np.int64), 'C': np.zeros((M, 4), np.int64)}
    out = {}
    for node in NODES:                      # upstream first: pending of node n is complete before n is simulated
        plan = bits[:, NODE_OFF[node]:NODE_OFF[node] + 4]
        for phase in PHASES:
            q = np.zeros(M, np.int64)
            dl = np.zeros(M, np.int64)
            ov = np.zeros(M, np.int64)
            green_val = 1 if phase == 'NS' else 0
            for t in range(4):
                avail = q + s['arrivals'][node][phase][t]
                if node in link_in and phase == link_in[node]['phase']:
                    avail = avail + pending[node][:, t]
                dep = np.where(plan[:, t] == green_val, np.minimum(avail, cap), 0)
                q = avail - dep
                dl += q
                ov += np.maximum(0, q - store)
                if node in link_out and phase == link_out[node]['phase'] and t < 3:
                    pending[link_out[node]['to']][:, t + 1] += lut[node][dep]
            out[(node, phase)] = (dl + w * ov).astype(float)
    sw = np.zeros(M)
    for node in NODES:
        plan = bits[:, NODE_OFF[node]:NODE_OFF[node] + 4]
        sw += (plan[:, 1:] != plan[:, :-1]).sum(1)
    out['switch'] = s['switch_weight'] * sw
    return out


def cost_table(s, bits=ALL_BITS):
    """Official chain cost of every schedule (row index = int(schedule string, 2))."""
    return np.round(sum(approach_tables(s, bits).values()), 6)


def scopes(s):
    """Light-cone scope S_k (sorted bit indices) of each approach; see module docstring."""
    a = s['links'][0]['phase']
    b = s['links'][1]['phase']
    rng = lambda node, slots: [NODE_OFF[node] + t for t in range(slots)]
    out = {}
    for ph in PHASES:
        out[('A', ph)] = rng('A', 4)
        out[('B', ph)] = rng('B', 4) + (rng('A', 3) if ph == a else [])
        if ph == b:
            out[('C', ph)] = rng('C', 4) + rng('B', 3) + (rng('A', 2) if b == a else [])
        else:
            out[('C', ph)] = rng('C', 4)
    return {k: sorted(v) for k, v in out.items()}


def lightcone_pairs(s):
    """Pairs (i<j) that can carry a coupling under the uniform-measure projection: pairs inside some scope.
    (Switching pairs (n,t),(n,t+1) lie inside every own-node scope.)"""
    pairs = set()
    for sc in scopes(s).values():
        pairs.update(itertools.combinations(sc, 2))
    return sorted(pairs)


# ---------------------------------------------------------------- scenarios
def random_chain_scenario(rng, name='chain'):
    """Chain scenario drawn from the official hidden ranges, applied to all three intersections and both links."""
    return {
        'name': name,
        'arrivals': {n: {p: [rng.randint(0, 8) for _ in range(4)] for p in PHASES} for n in NODES},
        'service_capacity': rng.randint(4, 6),
        'storage_capacity': rng.randint(5, 8),
        'overflow_weight': rng.choice([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]),
        'switch_weight': rng.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
        'links': [{'from': 'A', 'to': 'B', 'phase': rng.choice(PHASES), 'fraction': rng.choice([0.4, 0.5, 0.6, 0.7, 0.8])},
                  {'from': 'B', 'to': 'C', 'phase': rng.choice(PHASES), 'fraction': rng.choice([0.4, 0.5, 0.6, 0.7, 0.8])}],
    }


def random_chain_scenarios(n, seed, min_gap=5.0):
    """n chain scenarios with C_rand - C* > min_gap (same filter as traffic_model.random_scenarios)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = random_chain_scenario(rng, name=f'chain{seed}_{len(out)}')
        c = cost_table(s)
        if c.mean() - c.min() > min_gap:
            out.append(s)
    return out


def scenario_R3(R):
    """Illustrative corridor built from the public scenario R by the rule 'repeat the downstream intersection':
    A, B and the A->B link are R's; C receives B's arrival profile and a B->C link identical to R's A->B link."""
    return cs.embed_two_node(R, c_arrivals={p: list(R['arrivals']['B'][p]) for p in PHASES},
                             bc_phase=R['link']['phase'], bc_fraction=R['link']['fraction'], name='R3')
