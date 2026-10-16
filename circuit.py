"""Qiskit circuit construction, OpenQASM 3 export and final sampling."""
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

N = 8


def qubit_of(bit):
    """Schedule bit i (A1..A4, B1..B4) -> qubit 7 - i, so Qiskit keys read A1..B4 left to right."""
    return N - 1 - bit


def build_circuit(ansatz, params, measure=True):
    qc = QuantumCircuit(N, N if measure else 0, name='traffic_qaoa')
    qc.h(range(N))
    for gf, gj, b in ansatz.layer_angles(params):
        for i in range(N):
            if ansatz.h[i] != 0:
                qc.rz(2 * gf[i] * ansatz.h[i], qubit_of(i))
        for i in range(N):
            for j in range(i + 1, N):
                if ansatz.J[i, j] != 0:
                    qc.rzz(2 * gj * ansatz.J[i, j], qubit_of(i), qubit_of(j))
        for i in range(N):
            qc.rx(-2 * b[i], qubit_of(i))
    if measure:
        qc.barrier()
        for i in range(N):
            qc.measure(qubit_of(i), qubit_of(i))
    return qc


def circuit_probabilities(qc):
    """Exact outcome distribution of the (measurement-free part of the) circuit, keyed by schedule index."""
    sv = Statevector(qc.remove_final_measurements(inplace=False))
    return np.abs(sv.data) ** 2   # Qiskit index = sum_q bit_q 2^q = int(schedule string, 2)


def sample(qc, shots, seed):
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    result = sampler.run([qc]).result()[0]
    counts = result.data.c.get_counts()
    assert sum(counts.values()) == shots
    return dict(sorted(counts.items()))


def to_qasm3(qc):
    return qasm3.dumps(qc)
