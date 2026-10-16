"""n-qubit Qiskit circuit, OpenQASM 3 export and seeded 4,096-shot sampling (generalises circuit.py).

Schedule bit i is placed on qubit n-1-i and measured into clbit n-1-i, so Qiskit count keys (c_{n-1}..c_0)
read directly as schedule strings A1..A4 B1..B4 C1..C4 and the basis index is int(schedule, 2).
"""
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector


def build_circuit(ansatz, params, measure=True):
    n = ansatz.n
    qubit_of = lambda bit: n - 1 - bit
    qc = QuantumCircuit(n, n if measure else 0, name='chain_qaoa')
    qc.h(range(n))
    for gf, gj, b in ansatz.layer_angles(params):
        for i in range(n):
            if ansatz.h[i] != 0:
                qc.rz(2 * gf[i] * ansatz.h[i], qubit_of(i))
        for i in range(n):
            for j in range(i + 1, n):
                if ansatz.J[i, j] != 0:
                    qc.rzz(2 * gj * ansatz.J[i, j], qubit_of(i), qubit_of(j))
        for i in range(n):
            qc.rx(-2 * b[i], qubit_of(i))
    if measure:
        qc.barrier()
        for i in range(n):
            qc.measure(qubit_of(i), qubit_of(i))
    return qc


def circuit_probabilities(qc):
    sv = Statevector(qc.remove_final_measurements(inplace=False))
    return np.abs(sv.data) ** 2


def sample(qc, shots, seed):
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    result = sampler.run([qc]).result()[0]
    counts = result.data.c.get_counts()
    assert sum(counts.values()) == shots
    return dict(sorted(counts.items()))


def to_qasm3(qc):
    return qasm3.dumps(qc)
