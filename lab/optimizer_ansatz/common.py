"""Shared harness for the optimizer/ansatz lab (lab/optimizer_ansatz).

Nothing in the repo is modified: repo modules are imported from the repo root.
Problem sets
  TUNE(n)  = traffic_model.random_scenarios(n, seed=31)           (selection only)
  TUNE2(n) = traffic_model.random_scenarios(n, seed=7)            (second tuning seed, confirmation)
  VAL      = [scenario R] + traffic_model.random_scenarios(60, seed=555)   (reporting only)
"""
import os
import sys
import pickle
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import baselines as bl  # noqa: E402
import qubo as qb  # noqa: E402
import traffic_model as tm  # noqa: E402

CACHE = HERE / 'cache'
CACHE.mkdir(exist_ok=True)


class Problem:
    """Scenario + official cost table (benchmark only) + QUBO/Ising (training only)."""

    def __init__(self, s, costs, measure='adaptive', kappa=1.0):
        self.s, self.costs = s, np.asarray(costs, float)
        self.offset, self.Q = qb.build_qubo(s, measure=measure, kappa=kappa)
        self.h, self.J, self.const = qb.qubo_to_ising(self.Q, self.offset)
        self.E = qb.energies(self.Q, self.offset)
        self.name = s.get('name')


def _scen_set(key, maker):
    f = CACHE / f'{key}.pkl'
    if f.exists():
        return pickle.loads(f.read_bytes())
    scs = maker()
    data = [(s, tm.cost_table(s)) for s in scs]
    f.write_bytes(pickle.dumps(data))
    return data


def tune_set(n=40, seed=31):
    assert seed not in (2026, 99, 555)
    data = _scen_set(f'tune_s{seed}_n{n}', lambda: tm.random_scenarios(n, seed=seed))
    return [Problem(s, c) for s, c in data]


def val_set():
    def mk():
        R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
        return [R] + tm.random_scenarios(60, seed=555)
    data = _scen_set('val_R_s555_n60', mk)
    return data   # raw (scenario, costs) pairs; VAL is evaluated through the full solve() pipeline


def exact_metrics(prob, probs):
    m = bl.distribution_metrics(prob.costs, probs)
    return {'lift': m['lift'], 'enrichment': m['enrichment'], 'p_optimal': m['p_optimal'], 'E_Q': m['E_Q']}


def summarize(rows, keys=('lift', 'enrichment', 'p_optimal')):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        out[k] = (float(v.mean()), float(v.min()))
    return out


def fmt(summary):
    return '  '.join(f"{k}={m:.4f}(min {lo:.3f})" for k, (m, lo) in summary.items())
