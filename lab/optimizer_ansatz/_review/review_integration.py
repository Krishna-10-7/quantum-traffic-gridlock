"""Simulate the proposed integration in-process (no repo file is touched) and run the repo's own tests,
plus a QASM3 round-trip check and an ablation-row smoke test.

    python lab/optimizer_ansatz/_review/review_integration.py
"""
import os
import sys

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import importlib.util
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
ROOT = AGENT.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from qiskit import qasm3  # noqa: E402

import circuit as cq  # noqa: E402
import qaoa  # noqa: E402
import run  # noqa: E402
import traffic_model as tm  # noqa: E402

spec = importlib.util.spec_from_file_location('qaoa_nft', AGENT / 'qaoa_nft.py')
nft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nft)
ORIG = qaoa.train


def nft_train(*a, **k):
    # fallback path inside qaoa_nft calls qaoa.train: point it at the original
    qaoa.train = ORIG
    try:
        return nft.train(*a, **k)
    finally:
        qaoa.train = nft_train


qaoa.train = nft_train
run.DEFAULTS.update(cost_mode='field', stage1_evals=10)

spec_t = importlib.util.spec_from_file_location('tp', ROOT / 'tests' / 'test_pipeline.py')
tp = importlib.util.module_from_spec(spec_t)
spec_t.loader.exec_module(tp)
for name in dir(tp):
    if name.startswith('test_'):
        getattr(tp, name)()
        print('PASS (integrated candidate)', name, flush=True)

# QASM3 round trip on a few VAL scenarios
scs = [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(4, seed=555)
for s in scs:
    r = run.solve(s, 0)
    qc = qasm3.loads(r['qasm'])
    pq = cq.circuit_probabilities(qc)
    a = qaoa.Ansatz(np.array(r['qubo']['ising']['h']), np.array(r['qubo']['ising']['J_upper']), 4, 'field', 'qubit')
    pt = a.probabilities(np.array(r['training']['params']))
    emp = np.zeros(256)
    for kk, n in r['counts'].items():
        emp[int(kk, 2)] = n / 4096
    print(s['name'], 'qasm-vs-trainer', float(np.abs(pq - pt).max()), 'TV', 0.5 * float(np.abs(emp - pq).sum()),
          'evals', r['metrics']['training']['objective_evaluations'], 'ops', r['metrics']['circuit']['ops'], flush=True)

# ablation rows of experiments/benchmark.py still run with the integrated trainer
R = scs[0]
for name, cfg in [('standard', {'cost_mode': 'single', 'mixer_mode': 'single'}), ('per-qubit mixer only', {'cost_mode': 'single'}),
                  ('cvar', {'kind': 'cvar'}), ('energy', {'kind': 'energy'}), ('init random', {'init': 'random'}),
                  ('init small', {'init': 'small'}), ('budget 100', {'max_evals': 100}), ('p=1', {'p': 1}),
                  ('uniform QUBO', {'measure': 'uniform'})]:
    r = run.solve(R, 0, cfg)
    q = r['metrics']['baselines']['6_qaoa_raw_samples']
    print(f"ablation {name:22s} lift {q['lift']:.4f} enr {q['enrichment']:.2f} evals "
          f"{r['metrics']['training']['objective_evaluations']}", flush=True)
