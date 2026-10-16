"""Review: verify the offline learning of theta_bar.

1. BatchQAOA mean objective == mean of qaoa.make_objective('gibbs', eta=2) over the scenarios (repo code).
2. Adjoint gradient vs central finite differences of the repo objective.
3. Regenerate theta_bar from FRESH random_scenarios(150, seed=31) with the official cost tables replaced by
   zeros (so any dependence on official costs would show up), compare with the stored constant.
4. Concentration claim: median distance of cached per-instance optima to theta_bar, objective at their median.
Writes nothing outside lab/transfer_init/_review/.
"""
import os
import sys
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TI = ROOT / 'lab' / 'transfer_init'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TI))
import numpy as np

import qaoa
import qubo as qb
import traffic_model as tm
import common as C

stored = json.loads((TI / 'transfer_angles.json').read_text())['split_qubit_eta2']
theta = np.array(stored['theta'])
scen = tm.random_scenarios(150, seed=31)
probs = [C.Problem(s, np.zeros(256)) for s in scen]          # official costs deliberately zeroed


def repo_mean(x, idx=None):
    vals = []
    for pr in (probs if idx is None else [probs[i] for i in idx]):
        a = qaoa.Ansatz(pr.h, pr.J, 4, 'split', 'qubit')
        vals.append(qaoa.make_objective(a, pr.E, 'gibbs', 0.25, 2.0)(x))
    return float(np.mean(vals))


model = C.BatchQAOA(probs, 4, 'split', 'qubit', 'gibbs', 2.0)
print('1. batch mean f(theta_bar) =', model.value(theta), ' repo mean f =', repo_mean(theta),
      ' stored mean_objective =', stored['mean_objective'])

rng = np.random.default_rng(1)
x = rng.uniform(0, 1.5, 40)
sub = list(range(10))
m10 = C.BatchQAOA([probs[i] for i in sub], 4, 'split', 'qubit', 'gibbs', 2.0)
v, gA = m10.value_grad(x)
eps = 1e-5
gF = np.array([(repo_mean(x + eps * e, sub) - repo_mean(x - eps * e, sub)) / (2 * eps) for e in np.eye(40)])
print('2. adjoint vs FD gradient: max |diff| =', float(np.abs(gA - gF).max()), ' max |g| =', float(np.abs(gF).max()))
_, g0 = model.value_grad(theta)
print('   gradient norm at stored theta_bar (150 scen):', float(np.linalg.norm(g0)))

t = time.time()
best = None
for dt in (0.5, 0.75, 1.0, 1.25, 1.5):
    xs, vs = C.lbfgs(model, C.ramp(4, model.mg, model.mb, dt), maxiter=3000)
    print(f'   ramp {dt}: mean f {vs:.6f}')
    if best is None or vs < best[1] - 1e-9:
        best = (xs, vs)
xr = np.round(best[0], 6)
print(f'3. regenerated (fresh scenarios, zeroed costs) theta vs stored: max |diff| = {np.abs(xr - theta).max():.2e}, '
      f'mean f {best[1]:.6f} ({time.time() - t:.0f}s)')

opt = np.load(TI / 'cache' / 'optima_split_qubit_eta2.npy')
d = np.linalg.norm(opt - theta, axis=1)
print('4. optima shape', opt.shape, ' median dist to theta_bar', float(np.median(d)),
      ' mean f at coordinate median of optima', model.value(np.median(opt, 0)),
      ' mean f at theta_bar', model.value(theta))
