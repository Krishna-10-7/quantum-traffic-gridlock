"""Larger tuning sweep (seed=31, n=200): expansion point rules."""
import os, sys
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'): os.environ[_v]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1])); sys.path.insert(0,str(HERE))
import numpy as np, traffic_model as tm, baselines as bl, qubo as qb
from explore import surrogate_metrics, summarize, fmt
from explore2 import local_tables, build_node_measure, node_costs
from explore3 import build_pbiased

def p_cum(s, clip=0.1, sm=0.5, link=0.0, mode='cum'):
    a=s['arrivals']; f=s['link']['fraction']; lp=s['link']['phase']; p=np.zeros(8)
    for n,node in enumerate('AB'):
        ns=np.array(a[node]['NS'],float); ew=np.array(a[node]['EW'],float)
        if node=='B' and link:
            al=np.array(a['A'][lp],float); extra=np.r_[0,f*al[:3]]*link
            if lp=='NS': ns=ns+extra
            else: ew=ew+extra
        if mode=='cum': N,E=np.cumsum(ns),np.cumsum(ew)
        elif mode=='flat': N,E=np.full(4,ns.sum()),np.full(4,ew.sum())
        elif mode=='half': N,E=np.ones(4),np.ones(4)
        p[4*n:4*n+4]=(N+sm)/(N+E+2*sm)
    return np.clip(p,clip,1-clip)

if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 200
    scs=tm.random_scenarios(n, seed=31)
    res={}
    for s in scs:
        costs=tm.cost_table(s); tabs=local_tables(s); cands={}
        for meas in ('uniform','adaptive'):
            off,Q=qb.build_qubo(s,meas); cands['proj_'+meas]=qb.energies(Q,off)
        for lab,kw in [('cum c.1',{}),('cum c.05',dict(clip=0.05)),('cum c.2',dict(clip=0.2)),('cum c.3',dict(clip=0.3)),
                       ('cum sm2',dict(sm=2.0)),('cum link.5',dict(link=0.5)),('cum link1',dict(link=1.0)),
                       ('flat c.1',dict(mode='flat'))]:
            off,Q=build_pbiased(s,tabs,p_cum(s,**kw)); cands['P '+lab]=qb.energies(Q,off)
        cA,cBu,pats,vals,sw=node_costs(s,tabs); bidx=pats[:,3:]@[8,4,2,1]
        cB=cBu+np.array([vals[bidx==k].mean() for k in range(16)])+sw
        for kap in (0.5,1.0):
            wA=np.exp(-(cA-cA.min())/(kap*cA.std())); wB=np.exp(-(cB-cB.min())/(kap*cB.std()))
            off,Q=build_node_measure(s,tabs,wA,wB); cands[f'LB k{kap}']=qb.energies(Q,off)
        for k,E in cands.items(): res.setdefault(k,[]).append(surrogate_metrics(E,costs))
    for k,rows in res.items(): print(fmt(summarize(rows,k)))
