"""General L-intersection corridor n_0 -> n_1 -> ... -> n_{L-1} (scorer.py semantics), used ONLY to verify the
light-cone theorem and the cost of the approach-wise (local) QUBO construction beyond three intersections.

Link l (n_l -> n_{l+1}) has phase phi_l and fraction f_l. Light cone of approach (n_j, phi), T = 4 slots:
    S(n_j, phi) = { (n_j, 1..T) }  U  { (n_{j-m}, 1..T-m) : 1 <= m <= M },
where M is the largest m <= min(j, T-1) with phi_{j-1} = phi and phi_{j-1} = phi_{j-2} = ... = phi_{j-m}
(the vehicles must stay on the linked phase through every intermediate node). Hence |S| <= T(T+1)/2 = 10.

    python lab/chain_extension/corridor.py      # verifies the theorem for L = 4 (16 bits, 65,536 schedules)
"""
import itertools
import os
import random
import sys
import time
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import numpy as np  # noqa: E402

import chain_model as cm  # noqa: E402
import chain_qubo as cqb  # noqa: E402
import chain_scorer as cs  # noqa: E402

T = 4
PHASES = ('NS', 'EW')


def random_corridor(rng, L, aligned=False):
    ph = rng.choice(PHASES)
    return {'L': L,
            'arrivals': [{p: [rng.randint(0, 8) for _ in range(T)] for p in PHASES} for _ in range(L)],
            'service_capacity': rng.randint(4, 6), 'storage_capacity': rng.randint(5, 8),
            'overflow_weight': rng.choice([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]),
            'switch_weight': rng.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
            'links': [{'phase': ph if aligned else rng.choice(PHASES), 'fraction': rng.choice([0.4, 0.5, 0.6, 0.7, 0.8])}
                      for _ in range(L - 1)]}


def approach_tables(c, bits):
    bits = np.asarray(bits, np.int64)
    M, L = bits.shape[0], c['L']
    cap, store, w = c['service_capacity'], c['storage_capacity'], c['overflow_weight']
    pending = [np.zeros((M, T), np.int64) for _ in range(L)]
    out = {}
    for j in range(L):
        plan = bits[:, T * j:T * j + T]
        for ph in PHASES:
            q = np.zeros(M, np.int64)
            tot = np.zeros(M)
            linked_in = j > 0 and c['links'][j - 1]['phase'] == ph
            linked_out = j < L - 1 and c['links'][j]['phase'] == ph
            lut = (np.array([cs.transfer(c['links'][j]['fraction'], d) for d in range(cap + 1)]) if linked_out else None)
            for t in range(T):
                avail = q + c['arrivals'][j][ph][t] + (pending[j][:, t] if linked_in else 0)
                dep = np.where(plan[:, t] == (1 if ph == 'NS' else 0), np.minimum(avail, cap), 0)
                q = avail - dep
                tot += q + w * np.maximum(0, q - store)
                if linked_out and t < T - 1:
                    pending[j + 1][:, t + 1] += lut[dep]
            out[(j, ph)] = tot
    sw = sum((bits[:, T * j + 1:T * j + T] != bits[:, T * j:T * j + T - 1]).sum(1) for j in range(L))
    out['switch'] = c['switch_weight'] * sw.astype(float)
    return out


def scope(c, j, ph):
    s = [T * j + t for t in range(T)]
    m = 0
    while m < min(j, T - 1) and c['links'][j - 1 - m]['phase'] == ph:
        m += 1
        s += [T * (j - m) + t for t in range(T - m)]
    return sorted(s)


def to_chain(c):
    """L = 3 corridor as a chain_scorer scenario (for the cross-check)."""
    return {'name': 'corr', 'arrivals': dict(zip('ABC', c['arrivals'])),
            **{k: c[k] for k in ('service_capacity', 'storage_capacity', 'overflow_weight', 'switch_weight')},
            'links': [{'from': 'A', 'to': 'B', **c['links'][0]}, {'from': 'B', 'to': 'C', **c['links'][1]}]}


def local_uniform_qubo(c):
    """Approach-wise uniform-measure projection (2^|S_k| simulations per approach) + exact switching."""
    n = T * c['L']
    Q = np.zeros((n, n))
    off, sims = 0.0, 0
    for j in range(c['L']):
        for ph in PHASES:
            sc = scope(c, j, ph)
            b = cqb.Basis(len(sc))
            full = np.zeros((2 ** len(sc), n), np.int8)
            full[:, sc] = b.X.astype(np.int8)
            v = approach_tables(c, full)[(j, ph)]
            sims += 2 ** len(sc)
            o, Qk = b.to_qubo(cqb._project(b.F, v, np.ones(len(v))))
            off += o
            for a in range(len(sc)):
                for bb in range(a, len(sc)):
                    Q[sc[a], sc[bb]] += Qk[a, bb]
    w = c['switch_weight']
    for j in range(c['L']):
        for t in range(1, T):
            i = T * j + t
            Q[i, i] += w
            Q[i - 1, i - 1] += w
            Q[i - 1, i] -= 2 * w
    return off, Q, sims


def main():
    t0 = time.time()
    rng = random.Random(4)
    # (1) L = 3 corridor == chain model
    for _ in range(5):
        c = random_corridor(rng, 3)
        a = approach_tables(c, cm.ALL_BITS)
        assert np.allclose(sum(a.values()), cm.cost_table(to_chain(c)))
        assert all(scope(c, j, ph) == cm.scopes(to_chain(c))[('ABC'[j], ph)] for j in range(3) for ph in PHASES)
    print('L=3 corridor model == chain model (cost and scopes)')
    # (2) L = 4: light-cone invariance and tightness on all 65,536 schedules
    L = 4
    X = np.array(list(itertools.product([0, 1], repeat=T * L)), dtype=np.int8)
    used = {}
    for k in range(12):
        c = random_corridor(rng, L, aligned=(k % 2 == 0))
        tab = approach_tables(c, X)
        for j in range(L):
            for ph in PHASES:
                sc = scope(c, j, ph)
                v = tab[(j, ph)]
                key = X[:, sc].astype(np.int64) @ (1 << np.arange(len(sc)))
                lo = np.full(1 << len(sc), np.inf)
                hi = np.full(1 << len(sc), -np.inf)
                np.minimum.at(lo, key, v)
                np.maximum.at(hi, key, v)
                assert np.allclose(lo, hi), (j, ph)
                for i in sc:
                    if np.any(v != v[np.arange(len(X)) ^ (1 << (T * L - 1 - i))]):
                        used.setdefault((j, len(sc)), set()).add(i)
    assert max(len(scope(random_corridor(random.Random(0), L, aligned=True), 3, ph)) for ph in PHASES) == 10
    print('L=4 light-cone invariance verified on 65,536 schedules x 12 corridors; max |S| = 10 = T(T+1)/2')
    print('bits that actually matter, by (node, |S|):', {k: len(v) for k, v in sorted(used.items())})
    # (3) local construction == global uniform projection for L = 4, and its simulation count
    c = random_corridor(random.Random(5), L, aligned=True)
    off, Q, sims = local_uniform_qubo(c)
    total = sum(approach_tables(c, X).values())
    e_local = off + np.einsum('si,ij,sj->s', X.astype(float), Q, X.astype(float))
    z = 1 - 2 * X.astype(float)
    resid = total - e_local       # orthogonal to all Walsh features of order <= 2  <=>  equal to global projection
    worst = max(abs(resid.mean()), max(abs(resid @ z[:, i]) / len(X) for i in range(16)),
                max(abs(resid @ (z[:, i] * z[:, j])) / len(X) for i in range(16) for j in range(i + 1, 16)))
    assert worst < 1e-9, worst
    print(f'L=4 aligned corridor: local uniform QUBO uses {sims} simulations (global: {len(X)}); '
          f'residual orthogonal to all order<=2 Walsh features (max {worst:.1e})')
    for LL in range(2, 9):
        cc = random_corridor(random.Random(1), LL, aligned=True)
        print(f'  L={LL}: aligned-corridor local simulations = '
              f'{sum(2 ** len(scope(cc, j, ph)) for j in range(LL) for ph in PHASES)}  vs global 2^{T * LL}')
    print(f'done in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
