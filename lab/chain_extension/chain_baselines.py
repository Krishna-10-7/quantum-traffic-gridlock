"""Baselines and sampling metrics for the 12-bit chain (all costs from chain_scorer semantics).

Elite set: the 208 lowest-cost schedules (ties by ascending bitstring); 208 = round(4096 * 13/256) = round(0.0508 * 4096).
Enrichment = (shot fraction in the elite set) / (208/4096): uniform random sampling has expected enrichment 1 and the
maximum is 4096/208 = 19.69, the same scale as the official 13/256 definition.
"""
import itertools

import numpy as np

import chain_model as cm
import chain_scorer as cs

DIM = 2 ** cm.N
ELITE_SIZE = int(round(DIM * 13 / 256))   # 208


def alternating(s, costs):
    """Best of the 8 schedules in which every intersection alternates (1010 or 0101)."""
    rows = [(float(costs[int(''.join(c), 2)]), ''.join(c)) for c in itertools.product(('1010', '0101'), repeat=3)]
    c, bits = min(rows)
    return {'bits': bits, 'cost': c}


def greedy(s):
    """Per slot, each intersection (A, then B, then C) serves the phase with the larger queue after arrivals
    (ties -> NS), simulated with the chain dynamics (transfers arrive one slot later)."""
    cap = s['service_capacity']
    link_in = {l['to']: l for l in s['links']}
    link_out = {l['from']: l for l in s['links']}
    q = {n: {p: 0 for p in cm.PHASES} for n in cm.NODES}
    pending = {'B': [0] * 4, 'C': [0] * 4}
    plan = {n: [] for n in cm.NODES}
    for t in range(4):
        for node in cm.NODES:
            avail = {}
            for ph in cm.PHASES:
                extra = pending[node][t] if node in link_in and ph == link_in[node]['phase'] else 0
                avail[ph] = q[node][ph] + s['arrivals'][node][ph][t] + extra
            bit = 1 if avail['NS'] >= avail['EW'] else 0
            plan[node].append(bit)
            for ph in cm.PHASES:
                d = min(avail[ph], cap) if bit == (1 if ph == 'NS' else 0) else 0
                q[node][ph] = avail[ph] - d
                if node in link_out and ph == link_out[node]['phase'] and t < 3:
                    pending[link_out[node]['to']][t + 1] += cs.transfer(link_out[node]['fraction'], d)
    bits = ''.join(map(str, plan['A'] + plan['B'] + plan['C']))
    return {'bits': bits, 'cost': cs.evaluate(s, bits)['cost']}


def exact(costs):
    k = int(np.argmin(costs))
    return {'bits': cm.BITSTRINGS[k], 'cost': float(costs[k])}


def elite_set(costs):
    # lexsort: last key is primary -> (cost, index); index order == ascending bitstring order
    return np.lexsort((np.arange(len(costs)), costs))[:ELITE_SIZE]


def sample_metrics(costs, counts):
    shots = sum(counts.values())
    c_star, c_rand = float(costs.min()), float(costs.mean())
    idx = np.array([int(b, 2) for b in counts])
    n = np.array(list(counts.values()), dtype=float)
    eq = float((costs[idx] * n).sum() / shots)
    elite = np.zeros(len(costs), bool)
    elite[elite_set(costs)] = True
    fe = float(n[elite[idx]].sum() / shots)
    best = min(counts, key=lambda b: (costs[int(b, 2)], b))
    return {'shots': shots, 'E_Q': eq, 'C_star': c_star, 'C_rand': c_rand,
            'lift': (c_rand - eq) / (c_rand - c_star), 'elite_fraction': fe, 'enrichment': fe / (ELITE_SIZE / len(costs)),
            'p_optimal': float(n[costs[idx] == c_star].sum() / shots),
            'best_sampled': {'bits': best, 'cost': float(costs[int(best, 2)])}, 'distinct_bitstrings': len(counts)}


def distribution_metrics(costs, probs):
    c_star, c_rand = float(costs.min()), float(costs.mean())
    eq = float(probs @ costs)
    fe = float(probs[elite_set(costs)].sum())
    return {'E_Q': eq, 'lift': (c_rand - eq) / (c_rand - c_star), 'enrichment': fe / (ELITE_SIZE / len(costs)),
            'p_optimal': float(probs[costs == c_star].sum())}


def hybrid_local_search(counts, costs, top=8):
    """SEPARATELY LABELLED HYBRID: steepest-descent 1-flip local search on the official chain cost from the `top`
    most frequent raw QAOA samples. Not raw QAOA output."""
    seeds = sorted(counts, key=lambda b: (-counts[b], b))[:top]
    best, evals = None, 0
    for b in seeds:
        cur = b
        while True:
            nbrs = [cur[:i] + ('1' if cur[i] == '0' else '0') + cur[i + 1:] for i in range(len(cur))]
            evals += len(nbrs)
            nxt = min(nbrs, key=lambda x: (costs[int(x, 2)], x))
            if costs[int(nxt, 2)] < costs[int(cur, 2)]:
                cur = nxt
            else:
                break
        if best is None or (costs[int(cur, 2)], cur) < (costs[int(best, 2)], best):
            best = cur
    return {'bits': best, 'cost': float(costs[int(best, 2)]), 'official_cost_evaluations': evals,
            'label': 'HYBRID (QAOA samples + classical 1-flip descent); not raw QAOA'}
