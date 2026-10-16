"""Extension: noise analysis and noise-aware coupling sparsification.

Model (precise definition): the final bound circuit is transpiled to {rz, sx, x, cx}
(all-to-all connectivity, optimisation level 1). Every cx is followed by a 2-qubit
depolarizing channel with probability e2, every sx/x by a 1-qubit depolarizing channel
with probability e2/10 (rz is virtual and noiseless), and every measured bit is flipped
independently with probability r = e2. Output probabilities are computed exactly with
Aer's density-matrix method (no shot noise); lift/enrichment are then evaluated with the
official scorer exactly as for the noiseless runs.

Sparsification: couplings with |J_ij| < tau * max|J| are removed from the cost Hamiltonian
before training (fewer cx gates). Trained noiselessly, evaluated under noise.

    python experiments/noise.py            # writes results/noise.json and results/noise.md
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402
from qiskit import transpile  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: E402

import baselines as bl  # noqa: E402
import circuit as cq  # noqa: E402
import qaoa  # noqa: E402
import qubo as qb  # noqa: E402
import run  # noqa: E402
import traffic_model as tm  # noqa: E402

E2 = [0.0, 0.001, 0.003, 0.01, 0.02, 0.05]
SIM = AerSimulator(method='density_matrix')


def readout(probs, r):
    """Independent bit-flip readout error on all 8 bits (exact convolution)."""
    t = probs.reshape((2,) * 8)
    for ax in range(8):
        t0, t1 = np.take(t, 0, axis=ax), np.take(t, 1, axis=ax)
        t = np.stack([(1 - r) * t0 + r * t1, r * t0 + (1 - r) * t1], axis=ax)
    return t.reshape(256)


def noisy_probs(qc, e2):
    tq = transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1, seed_transpiler=0)
    if e2 == 0:
        return cq.circuit_probabilities(tq), tq
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(e2 / 10, 1), ['sx', 'x'])
    nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ['cx'])
    c = tq.copy()
    c.save_probabilities()
    p = np.asarray(SIM.run(c, noise_model=nm).result().data()['probabilities'])
    return readout(p, e2), tq


def trained_circuit(s, p, tau=0.0):
    cfg = {**run.DEFAULTS, 'p': p}
    off, Q = qb.build_qubo(s, measure=cfg['measure'], kappa=cfg['kappa'])
    if tau > 0:
        off_diag = np.triu(np.abs(Q), 1)
        Q = np.where(np.triu(np.ones_like(Q), 1).astype(bool) & (off_diag < tau * off_diag.max()), 0.0, Q)
    a, x, _, _ = run.train_qubo(off, Q, cfg, seed=0)
    return cq.build_circuit(a, x, measure=False)


def main():
    scs = [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(8, seed=4242)
    variants = [(f'p={p}', p, 0.0) for p in (1, 2, 3, 4)] + [('p=4, pruned tau=0.1', 4, 0.1), ('p=4, pruned tau=0.25', 4, 0.25),
                                                             ('p=2, pruned tau=0.25', 2, 0.25)]
    out = {'E2': E2, 'variants': []}
    for name, p, tau in variants:
        lifts, enrs, cxs = np.zeros((len(scs), len(E2))), np.zeros((len(scs), len(E2))), []
        for si, s in enumerate(scs):
            costs = tm.cost_table(s)
            qc = trained_circuit(s, p, tau)
            for ei, e2 in enumerate(E2):
                pr, tq = noisy_probs(qc, e2)
                m = bl.distribution_metrics(costs, pr)
                lifts[si, ei], enrs[si, ei] = m['lift'], m['enrichment']
            cxs.append(tq.count_ops().get('cx', 0))
        v = {'name': name, 'p': p, 'tau': tau, 'cx_count_mean': float(np.mean(cxs)),
             'R_lift': lifts[0].tolist(), 'mean_lift': lifts.mean(0).tolist(), 'mean_enrichment': enrs.mean(0).tolist()}
        out['variants'].append(v)
        print(f"{name:24s} cx={v['cx_count_mean']:5.1f} lift " + ' '.join(f'{x:.3f}' for x in v['mean_lift']), flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'noise.json').write_text(json.dumps(out, indent=2))
    lines = ['# Noise analysis (R + 8 random scenarios, exact density-matrix probabilities)', '',
             'Mean lift vs two-qubit depolarizing rate e2 (1q rate e2/10, readout flip e2).', '',
             '| variant | cx | ' + ' | '.join(f'e2={e}' for e in E2) + ' |', '|---|---|' + '---|' * len(E2)]
    for v in out['variants']:
        lines.append(f"| {v['name']} | {v['cx_count_mean']:.0f} | " + ' | '.join(f'{x:.3f}' for x in v['mean_lift']) + ' |')
    lines += ['', 'Mean enrichment:', '', '| variant | ' + ' | '.join(f'e2={e}' for e in E2) + ' |', '|---|' + '---|' * len(E2)]
    for v in out['variants']:
        lines.append(f"| {v['name']} | " + ' | '.join(f'{x:.2f}' for x in v['mean_enrichment']) + ' |')
    (ROOT / 'results' / 'noise.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
