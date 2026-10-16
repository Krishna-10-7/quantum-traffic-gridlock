"""Required baselines and sampling-quality metrics (all evaluated with the official scorer)."""
import itertools

import numpy as np

import scorer
import traffic_model as tm

ELITE_SIZE = 13


def official(s, bitstring):
    return scorer.evaluate(s, [int(c) for c in bitstring])['cost']


def alternating(s):
    """Best of the four schedules where each intersection alternates (1010 or 0101)."""
    rows = []
    for a, b in itertools.product(('1010', '0101'), repeat=2):
        rows.append((official(s, a + b), a + b))
    c, bits = min(rows)
    return {'bits': bits, 'cost': c, 'all': {b: c for c, b in rows}}


def greedy(s):
    """Per slot, each intersection serves the approach with the larger queue after arrivals (ties -> NS).

    Simulated with the same dynamics as the scorer (transfers arrive at B in the next slot).
    """
    cap, frac, lp = s['service_capacity'], s['link']['fraction'], s['link']['phase']
    q = {n: {p: 0 for p in tm.PHASES} for n in tm.NODES}
    pending = [0] * 4
    plan = {'A': [], 'B': []}
    for t in range(4):
        for node in tm.NODES:
            avail = {}
            for ph in tm.PHASES:
                inflow = s['arrivals'][node][ph][t] + (pending[t] if node == 'B' and ph == lp else 0)
                avail[ph] = q[node][ph] + inflow
            bit = 1 if avail['NS'] >= avail['EW'] else 0
            plan[node].append(bit)
            for ph in tm.PHASES:
                green = bit == (1 if ph == 'NS' else 0)
                d = min(avail[ph], cap) if green else 0
                q[node][ph] = avail[ph] - d
                if node == 'A' and ph == lp and t < 3:
                    pending[t + 1] += tm.transfer(frac, d)
    bits = ''.join(map(str, plan['A'] + plan['B']))
    return {'bits': bits, 'cost': official(s, bits)}


def exact(costs):
    k = int(np.argmin(costs))
    return {'bits': tm.BITSTRINGS[k], 'cost': float(costs[k])}


def elite_set(costs):
    order = sorted(range(256), key=lambda i: (costs[i], tm.BITSTRINGS[i]))
    return order[:ELITE_SIZE]


def sample_metrics(costs, counts):
    """Judging metrics for a counts dict keyed by 8-bit schedule strings."""
    shots = sum(counts.values())
    c_star, c_rand = float(costs.min()), float(costs.mean())
    eq = sum(costs[int(b, 2)] * n for b, n in counts.items()) / shots
    elite = {tm.BITSTRINGS[i] for i in elite_set(costs)}
    frac_elite = sum(n for b, n in counts.items() if b in elite) / shots
    opt = {tm.BITSTRINGS[i] for i in range(256) if costs[i] == c_star}
    best = min(counts, key=lambda b: (costs[int(b, 2)], b))
    return {
        'shots': shots,
        'E_Q': eq,
        'C_star': c_star,
        'C_rand': c_rand,
        'lift': (c_rand - eq) / (c_rand - c_star) if c_rand > c_star else float('nan'),
        'elite_fraction': frac_elite,
        'enrichment': frac_elite / (ELITE_SIZE / 256),
        'p_optimal': sum(n for b, n in counts.items() if b in opt) / shots,
        'best_sampled': {'bits': best, 'cost': float(costs[int(best, 2)])},
        'distinct_bitstrings': len(counts),
    }


def distribution_metrics(costs, probs):
    """Same metrics for an exact probability vector (infinite-shot limit)."""
    c_star, c_rand = float(costs.min()), float(costs.mean())
    eq = float(probs @ costs)
    fe = float(probs[elite_set(costs)].sum())
    return {'E_Q': eq, 'lift': (c_rand - eq) / (c_rand - c_star), 'enrichment': fe / (ELITE_SIZE / 256),
            'p_optimal': float(probs[costs == c_star].sum())}


def hybrid_local_search(s, counts, costs, top=8):
    """SEPARATELY LABELLED HYBRID: take the `top` most frequent QAOA samples and apply
    steepest-descent single-bit-flip local search on the official cost. Not raw QAOA output."""
    seeds = sorted(counts, key=lambda b: (-counts[b], b))[:top]
    best = None
    evals = 0
    for b in seeds:
        cur = b
        while True:
            nbrs = [cur[:i] + ('1' if cur[i] == '0' else '0') + cur[i + 1:] for i in range(8)]
            evals += 8
            nxt = min(nbrs, key=lambda x: (costs[int(x, 2)], x))
            if costs[int(nxt, 2)] < costs[int(cur, 2)]:
                cur = nxt
            else:
                break
        if best is None or (costs[int(cur, 2)], cur) < (costs[int(best, 2)], best):
            best = cur
    return {'bits': best, 'cost': float(costs[int(best, 2)]), 'official_cost_evaluations': evals,
            'label': 'HYBRID (QAOA samples + classical 1-flip descent); not raw QAOA'}
