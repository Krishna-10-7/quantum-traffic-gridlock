"""Verification of the chain extension.  Run:  python lab/chain_extension/test_chain.py"""
import itertools
import os
import random
import sys
import time
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

import numpy as np  # noqa: E402

import chain_baselines as cb  # noqa: E402
import chain_circuit as cc  # noqa: E402
import chain_model as cm  # noqa: E402
import chain_qaoa as cq  # noqa: E402
import chain_qubo as cqb  # noqa: E402
import chain_run  # noqa: E402
import chain_scorer as cs  # noqa: E402
import qaoa  # noqa: E402
import qubo as qb  # noqa: E402
import scorer  # noqa: E402
import traffic_model as tm  # noqa: E402

R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
SW = np.array([sum(b[t] != b[t - 1] for t in range(1, 4)) for b in itertools.product([0, 1], repeat=4)])  # switches of 4 bits


def _official_scenarios(n, seed):
    rng = random.Random(seed)
    return [R] + [tm.random_scenario(rng, name=f'o{k}') for k in range(n)]


def test_reduction_to_official_AB():
    """C with zero arrivals and f_BC = 0  =>  C3(xA, xB, xC) = scorer(xA xB) + w_sw * sw(xC) for ALL 4096 schedules.
    In particular C3(xA, xB, 0000) = C3(xA, xB, 1111) = official cost (exact equality)."""
    rng = random.Random(11)
    for s2 in _official_scenarios(30, 5):
        s3 = cs.embed_two_node(s2, bc_phase=rng.choice(cm.PHASES), bc_fraction=0.0)
        assert cs.restrict_AB(s3)['arrivals'] == s2['arrivals']
        c2 = np.array([scorer.evaluate(s2, [int(v) for v in row])['cost'] for row in tm.ALL_BITS])  # official scorer
        expect = (c2[:, None] + s2['switch_weight'] * SW[None, :]).ravel()                       # index = 16*ab + c
        c3 = cm.cost_table(s3)
        assert np.allclose(c3, expect, atol=1e-9)
        for k in rng.sample(range(4096), 200):     # reference (pure-python) chain scorer as well
            assert abs(cs.evaluate(s3, cm.BITSTRINGS[k])['cost'] - expect[k]) < 1e-9
        for ab in range(256):
            for c in ('0000', '1111'):
                assert c3[int(tm.BITSTRINGS[ab] + c, 2)] == c2[ab]
    # official verification vectors of the problem statement, through the chain scorer
    s3 = cs.embed_two_node(R)
    for bits, cost in [('10101010', 116.0), ('11111111', 146.0), ('00000000', 237.0), ('01111001', 92.5)]:
        assert cs.evaluate(s3, bits + '0000')['cost'] == cost


def test_reduction_to_official_BC():
    """A with zero arrivals and f_AB = 0  =>  C3 = scorer(B->A, C->B renamed)(xB xC) + w_sw * sw(xA):
    the B->C link uses exactly the official link semantics."""
    rng = random.Random(12)
    for s2 in _official_scenarios(20, 6):
        s3 = {'name': 'bc', 'arrivals': {'A': {'NS': [0] * 4, 'EW': [0] * 4}, 'B': s2['arrivals']['A'], 'C': s2['arrivals']['B']},
              **{k: s2[k] for k in ('service_capacity', 'storage_capacity', 'overflow_weight', 'switch_weight')},
              'links': [{'from': 'A', 'to': 'B', 'phase': rng.choice(cm.PHASES), 'fraction': 0.0},
                        {'from': 'B', 'to': 'C', 'phase': s2['link']['phase'], 'fraction': s2['link']['fraction']}]}
        s2r = cs.restrict_BC(s3)
        c2 = np.array([scorer.evaluate(s2r, [int(v) for v in row])['cost'] for row in tm.ALL_BITS])
        expect = (s2['switch_weight'] * SW[:, None] + c2[None, :]).ravel()                        # index = 256*a + bc
        assert np.allclose(cm.cost_table(s3), expect, atol=1e-9)


def test_vectorised_equals_reference():
    for s in [cm.scenario_R3(R)] + cm.random_chain_scenarios(4, seed=7):
        ref = np.array([cs.evaluate(s, b)['cost'] for b in cm.BITSTRINGS])
        assert np.array_equal(ref, cm.cost_table(s))


def test_light_cone_scopes():
    """Every approach term is constant along bits outside its light-cone scope (checked on all 4096 inputs);
    every bit inside a scope matters in at least one test scenario (scopes are tight)."""
    used = {}
    for s in [cm.scenario_R3(R)] + cm.random_chain_scenarios(60, seed=8):
        tab = cm.approach_tables(s)
        for k, sc in cm.scopes(s).items():
            outside = [i for i in range(12) if i not in sc]
            key = cm.ALL_BITS[:, sc] @ (1 << np.arange(len(sc)))
            v = tab[k]
            lo = np.full(1 << len(sc), np.inf)
            hi = np.full(1 << len(sc), -np.inf)
            np.minimum.at(lo, key, v)
            np.maximum.at(hi, key, v)
            assert np.allclose(lo, hi), (k, outside)
            aligned = s['links'][0]['phase'] == s['links'][1]['phase']
            tag = (k[0], 'linked' if (k[0] == 'B' and k[1] == s['links'][0]['phase'])
                   or (k[0] == 'C' and k[1] == s['links'][1]['phase']) else 'unlinked', aligned if k[0] == 'C' else None)
            for i in sc:          # does flipping bit i ever change D_k ?
                flip = np.arange(4096) ^ (1 << (11 - i))
                if np.any(v != v[flip]):
                    used.setdefault(tag, set()).add(i)
    assert used[('C', 'linked', True)] == {0, 1, 4, 5, 6, 8, 9, 10, 11}
    assert used[('C', 'linked', False)] == {4, 5, 6, 8, 9, 10, 11}
    assert used[('B', 'linked', None)] == {0, 1, 2, 4, 5, 6, 7}


def test_local_uniform_equals_global_uniform():
    for s in [cm.scenario_R3(R)] + cm.random_chain_scenarios(6, seed=9):
        o1, Q1, i1 = cqb.build_qubo(s, measure='uniform')
        o2, Q2, i2 = cqb.build_qubo_local_uniform(s)
        assert abs(o1 - o2) < 1e-7 and np.allclose(Q1, Q2, atol=1e-7)
        assert i2['simulations'] < 4096
        # couplings only inside light-cone scopes
        allowed = set(cm.lightcone_pairs(s))
        nz = {(i, j) for i in range(12) for j in range(i + 1, 12) if Q1[i, j] != 0}
        assert nz <= allowed


def test_uniform_is_walsh_truncation_and_ising():
    s = cm.scenario_R3(R)
    c = cm.cost_table(s)
    off, Q, _ = cqb.build_qubo(s, measure='uniform')
    e = cqb.energies(Q, off)
    z = 1 - 2 * cqb.bit_matrix(12)
    resid = c - e
    assert abs(resid.mean()) < 1e-8
    for i in range(12):
        assert abs(resid @ z[:, i]) < 1e-6
        for j in range(i + 1, 12):
            assert abs(resid @ (z[:, i] * z[:, j])) < 1e-6
    for measure in ('uniform', 'adaptive'):
        off, Q, _ = cqb.build_qubo(s, measure=measure)
        h, J, const = cqb.qubo_to_ising(Q, off)
        assert np.allclose(const + z @ h + np.einsum('si,ij,sj->s', z, J, z), cqb.energies(Q, off))


def test_nqubit_trainer_reduces_to_repo_trainer():
    """With n = 8 the generalised ansatz/trainer is the repo's qaoa.py, bit for bit."""
    off, Q = qb.build_qubo(R)
    h, J, _ = qb.qubo_to_ising(Q, off)
    e = qb.energies(Q, off)
    rng = np.random.default_rng(0)
    for cm_, mm in [('single', 'single'), ('split', 'qubit'), ('field', 'qubit')]:
        a8, aN = qaoa.Ansatz(h, J, 4, cm_, mm), cq.Ansatz(h, J, 4, cm_, mm)
        x = rng.uniform(0, 1, a8.n_params)
        assert np.allclose(a8.probabilities(x), aN.probabilities(x), atol=1e-14)
    _, x8, h8 = qaoa.train(h, J, e, p=4, cost_mode='split', mixer_mode='qubit')
    _, xN, hN = cq.train(h, J, e, p=4, cost_mode='split', mixer_mode='qubit')
    assert len(h8) == len(hN) == 300 and np.allclose(h8, hN, atol=1e-12) and np.allclose(x8, xN, atol=1e-12)


def test_circuit_matches_trainer_and_bit_order():
    s = cm.scenario_R3(R)
    off, Q, _ = cqb.build_qubo(s)
    h, J, _ = cqb.qubo_to_ising(Q, off)
    a = cq.Ansatz(h, J, 2, 'split', 'qubit')
    x = np.random.default_rng(1).uniform(0, 1, a.n_params)
    qc = cc.build_circuit(a, x)
    assert np.abs(cc.circuit_probabilities(qc) - a.probabilities(x)).max() < 1e-10
    # bit order: a circuit that prepares schedule string b deterministically must return key b
    from qiskit import QuantumCircuit
    b = '110100101110'
    qc2 = QuantumCircuit(12, 12)
    for i, ch in enumerate(b):
        if ch == '1':
            qc2.x(11 - i)
    qc2.measure(range(12), range(12))
    assert cc.sample(qc2, 16, 0) == {b: 16}


def test_pipeline_budget_shots_determinism():
    s = cm.scenario_R3(R)
    r1 = chain_run.solve_chain(s, 3)
    r2 = chain_run.solve_chain(s, 3)
    m = r1['metrics']
    assert sum(r1['counts'].values()) == 4096 and r1['counts'] == r2['counts']
    assert m['training']['objective_evaluations'] <= 300 and m['circuit']['qubits'] == 12
    assert all(len(k) == 12 for k in r1['counts'])
    assert cb.ELITE_SIZE == 208


if __name__ == '__main__':
    t0 = time.time()
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            t = time.time()
            fn()
            print(f'PASS {name}  ({time.time() - t:.1f}s)')
    print(f'all tests passed in {time.time() - t0:.1f}s')
