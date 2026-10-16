import os, sys
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'): os.environ[_v]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1])); sys.path.insert(0,str(HERE))
import numpy as np, traffic_model as tm, qubo as qb
from analytic_qubo import build_analytic_qubo
from explore import build_mobius
from explore2 import local_tables
from explore3 import build_pbiased
from explore4 import p_cum
R=tm.load_scenario(HERE.parents[1]/'scenarios'/'scenario_R.json')
for s in [R]+tm.random_scenarios(20, seed=31):
    o,Q,info=build_analytic_qubo(s,'uniform',return_info=True); o2,Q2=qb.build_qubo(s,'uniform')
    assert np.allclose(qb.energies(Q,o),qb.energies(Q2,o2)), 'uniform mismatch'
    o,Q=build_analytic_qubo(s,'webster'); o2,Q2=build_pbiased(s,local_tables(s),p_cum(s,clip=0.3))
    assert np.allclose(qb.energies(Q,o),qb.energies(Q2,o2)), 'webster mismatch'
    o,Q=build_analytic_qubo(s,'red'); o2,Q2,_=build_mobius(s,'red')
    assert np.allclose(qb.energies(Q,o),qb.energies(Q2,o2)), 'red mismatch'
print('ok; approach sims', info)
o,Q=build_analytic_qubo(R,'webster'); E=qb.energies(Q,o); c=tm.cost_table(R)
k=int(np.argmin(E)); print('R webster min', tm.BITSTRINGS[k], c[k], 'C*', c.min(), 'pearson', np.corrcoef(E,c)[0,1])
np.set_printoptions(precision=2, suppress=True, linewidth=150); print(Q); print(o)
