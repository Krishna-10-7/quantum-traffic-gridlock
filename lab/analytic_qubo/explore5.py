"""Expansion-point refinements on tuning set (seed=31, n=200): Webster grid + mean-field (surrogate-Gibbs marginals)."""
import os, sys
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'): os.environ[_v]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1])); sys.path.insert(0,str(HERE))
import numpy as np, traffic_model as tm, qubo as qb
import analytic_qubo as aq
from explore import surrogate_metrics, summarize, fmt

def webster(s, clip=0.3, link=0.0, smooth=0.5):
    a=s['arrivals']; f=s['link']['fraction']; lp=s['link']['phase']; p=np.zeros(8)
    for n,node in enumerate('AB'):
        ns=np.array(a[node]['NS'],float); ew=np.array(a[node]['EW'],float)
        if node=='B' and link:
            al=np.array(a['A'][lp],float); extra=np.r_[0,f*al[:3]]*link
            if lp=='NS': ns=ns+extra
            else: ew=ew+extra
        N,E=np.cumsum(ns),np.cumsum(ew); p[4*n:4*n+4]=(N+smooth)/(N+E+2*smooth)
    return np.clip(p,clip,1-clip)

def build(comps, s, p8):
    Q=np.zeros((8,8)); off=0.0
    for name,vars_,pats,vals in comps:
        p=p8[vars_]; F0,V,W=aq.taylor_coefficients(pats,vals,p); off+=aq._add_taylor(Q,vars_,F0,V,W,p)
    aq._add_switching(s,Q); return off,Q

def meanfield(comps, s, p0, kappa=1.0, iters=8, clip=0.2):
    p=p0.copy()
    for _ in range(iters):
        off,Q=build(comps,s,p); E=qb.energies(Q,off)
        w=np.exp(-(E-E.min())/(kappa*E.std())); w/=w.sum()
        p=np.clip(w@qb.X, clip, 1-clip)
    return build(comps,s,p)

if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 200
    scs=tm.random_scenarios(n, seed=31); res={}
    for s in scs:
        costs=tm.cost_table(s); comps=aq.local_components(s); cands={}
        off,Q=qb.build_qubo(s,'adaptive'); cands['proj_adaptive']=qb.energies(Q,off)
        for clip in (0.25,0.3,0.35,0.4):
            for link in (0.0,1.0):
                off,Q=build(comps,s,webster(s,clip,link)); cands[f'web c{clip} l{link}']=qb.energies(Q,off)
        for kap in (1.0,2.0):
            for clip in (0.2,0.3):
                off,Q=meanfield(comps,s,webster(s,0.3),kap,clip=clip); cands[f'mf k{kap} c{clip}']=qb.energies(Q,off)
        for k,E in cands.items(): res.setdefault(k,[]).append(surrogate_metrics(E,costs))
    for k,rows in res.items(): print(fmt(summarize(rows,k)))
