import os, sys
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'): os.environ[_v]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1])); sys.path.insert(0,str(HERE))
import numpy as np, traffic_model as tm, baselines as bl, qubo as qb
from explore import *
scs=tm.random_scenarios(60, seed=31)
from collections import Counter
kNS=Counter(); dg=Counter(); kA=[]
for s in scs:
    c=tm.cost_table(s); k=int(np.argmin(c)); b=tm.BITSTRINGS[k]
    kNS[(b[:4].count('1'), b[4:].count('1'))]+=1
    g=bl.greedy(s)['bits']; dg[sum(x!=y for x,y in zip(b,g))]+=1
print('NS-green counts (A,B) at optimum', sorted(kNS.items()))
print('Hamming(opt, greedy)', sorted(dg.items()))
# load: per approach total arrivals / capacity
loads=[]
for s in scs:
    for n in 'AB':
        for p in ('NS','EW'):
            loads.append(sum(s['arrivals'][n][p])/s['service_capacity'])
print('approach demand in units of green slots: mean %.2f, quantiles'%np.mean(loads), np.quantile(loads,[.1,.5,.9]))
