"""Independent verification. Run: python tests/test_pipeline.py  (or pytest tests/)"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import baselines as bl  # noqa: E402
import circuit as cq  # noqa: E402
import qubo as qb  # noqa: E402
import run  # noqa: E402
import scorer  # noqa: E402
import traffic_model as tm  # noqa: E402

R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')


def test_official_reference_vectors():
    for bits, cost in [('10101010', 116.0), ('11111111', 146.0), ('00000000', 237.0), ('01111001', 92.5)]:
        assert scorer.evaluate(R, [int(c) for c in bits])['cost'] == cost


def test_decomposition_matches_scorer():
    rng = random.Random(3)
    for s in [R] + [tm.random_scenario(rng) for _ in range(30)]:
        for k in rng.sample(range(256), 40):
            bits = list(tm.ALL_BITS[k])
            assert abs(tm.cost(s, bits) - scorer.evaluate(s, [int(b) for b in bits])['cost']) < 1e-9


def test_locality_of_approach_costs():
    """A approaches depend only on A bits; the unlinked B approach only on B bits."""
    t = qb.approach_tables(R)
    unlinked = 'NS' if R['link']['phase'] == 'EW' else 'EW'
    grid = lambda v: v.reshape(16, 16)  # rows = A bits, cols = B bits
    for ph in tm.PHASES:
        assert np.allclose(grid(t[('A', ph)]), grid(t[('A', ph)])[:, :1])
    assert np.allclose(grid(t[('B', unlinked)]), grid(t[('B', unlinked)])[:1, :])


def test_qubo_ising_equivalence():
    for measure in ('uniform', 'adaptive'):
        off, Q = qb.build_qubo(R, measure=measure)
        h, J, const = qb.qubo_to_ising(Q, off)
        z = 1 - 2 * qb.X
        e_ising = const + z @ h + np.einsum('si,ij,sj->s', z, J, z)
        assert np.allclose(e_ising, qb.energies(Q, off))


def test_uniform_projection_is_walsh_truncation():
    """Uniform-measure fit == keep Walsh coefficients of order <= 2 (orthogonality check)."""
    c = tm.cost_table(R)
    off, Q = qb.build_qubo(R, measure='uniform')
    resid = c - qb.energies(Q, off)
    z = 1 - 2 * qb.X
    for i in range(8):
        assert abs(resid @ z[:, i]) < 1e-6
        for j in range(i + 1, 8):
            assert abs(resid @ (z[:, i] * z[:, j])) < 1e-6


def test_greedy_matches_published_vector():
    assert bl.greedy(R) == {'bits': '01111001', 'cost': 92.5}


def test_pipeline_consistency_and_determinism(tmp=ROOT / 'outputs' / '_test'):
    a = run.solve(R, seed=7)
    b = run.solve(R, seed=7)
    assert a['counts'] == b['counts']
    assert sum(a['counts'].values()) == 4096
    assert a['metrics']['training']['objective_evaluations'] <= 300
    assert a['metrics']['circuit']['qubits'] <= 16 and a['metrics']['circuit']['p'] <= 4
    # QASM round trip reproduces the exact distribution
    from qiskit import qasm3
    qc = qasm3.loads(a['qasm'])
    p_qasm = cq.circuit_probabilities(qc)
    off, Q = qb.build_qubo(R)
    assert p_qasm.shape == (256,)
    # counts are statistically consistent with the circuit (total variation distance)
    emp = np.zeros(256)
    for k, n in a['counts'].items():
        emp[int(k, 2)] = n / 4096
    tv = 0.5 * np.abs(emp - p_qasm).sum()
    assert tv < 0.06, tv
    # decoded schedule is scored correctly
    sch = a['schedule']
    assert scorer.evaluate(R, {'bits': sch['schedule']})['cost'] == sch['official_cost']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
