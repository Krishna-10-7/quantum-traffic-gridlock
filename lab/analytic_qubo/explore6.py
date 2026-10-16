"""Sharper Gibbs proxies (eta=6,10) for QAOA-like concentration; tuning set seed=31."""
import os, sys
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'): os.environ[_v]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1])); sys.path.insert(0,str(HERE))
import numpy as np, traffic_model as tm, qubo as qb, baselines as bl
import analytic_qubo as aq
from explore5 import build, meanfield

def prox(E,costs,eta):
    w=np.exp(-eta*(E-E.min())/E.std()); p=w/w.sum(); return bl.distribution_metrics(costs,p)

def hybrid(comps, s, p8, kappa=1.0):
    """per-node measure = Bernoulli(p*) x local Gibbs of the node's own cost"""
    P4=comps[0][2]; swc=s['switch_weight']*np.array([sum(p[t]!=p[t-1] for t in range(1,4)) for p in P4])
    cA=comps[0][3]+comps[1][3]+swc; pats7,vals7=comps[3][2],comps[3][3]; bidx=pats7[:,3:]@[8,4,2,1]
    cB=comps[2][3]+np.array([vals7[bidx==m].mean() for m in range(16)])+swc
    bern=lambda p: np.prod(np.where(P4==1,p,1-p),axis=1)
    wA=bern(p8[:4])*np.exp(-(cA-cA.min())/(kappa*cA.std())); wB=bern(p8[4:])*np.exp(-(cB-cB.min())/(kappa*cB.std()))
    Q=np.zeros((8,8)); off=0.0
    for name,vars_,pats,vals in comps:
        if name.startswith('A_'): w=wA[pats@[8,4,2,1]]
        elif name.endswith('linked'): w=wA.reshape(8,2).sum(1)[pats[:,:3]@[4,2,1]]*wB[pats[:,3:]@[8,4,2,1]]
        else: w=wB[pats@[8,4,2,1]]
        coef,pairs=aq._wls(pats,vals,w); k=len(vars_); off+=coef[0]
        for m,i in enumerate(vars_): Q[i,i]+=coef[1+m]
        for n_,(a,b) in enumerate(pairs):
            i,j=vars_[a],vars_[b]; Q[min(i,j),max(i,j)]+=coef[1+k+n_]
    aq._add_switching(s,Q); return off,Q

if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 200
    scs=tm.random_scenarios(n, seed=31); res={}
    for s in scs:
        costs=tm.cost_table(s); comps=aq.local_components(s); c={}
        c['adaptive']=qb.build_qubo(s,'adaptive'); c['uniform']=qb.build_qubo(s,'uniform')
        c['webster']=aq.build_analytic_qubo(s); c['local_gibbs']=aq.build_analytic_qubo(s,'local_gibbs')
        c['mf k1 c.2']=meanfield(comps,s,aq.webster_point(s),1.0,clip=0.2)
        c['hyb k1']=hybrid(comps,s,aq.webster_point(s),1.0); c['hyb k2']=hybrid(comps,s,aq.webster_point(s),2.0)
        for k,(o,Q) in c.items():
            E=qb.energies(Q,o); row={}
            for eta in (6,10):
                m=prox(E,costs,eta); row[f'lift{eta}']=m['lift']; row[f'enr{eta}']=m['enrichment']; row[f'popt{eta}']=m['p_optimal']
            kk=int(np.argmin(E)); row['reg']=(costs[kk]-costs.min())/costs.min()
            top=np.argsort(costs)[:30]; rk=lambda v: np.argsort(np.argsort(v)); row['sp_top30']=np.corrcoef(rk(E[top]),rk(costs[top]))[0,1]
            res.setdefault(k,[]).append(row)
    for k,rows in res.items():
        m={kk:np.mean([r[kk] for r in rows]) for kk in rows[0]}
        print(f"{k:<12} "+' '.join(f"{kk} {v:.4f}" for kk,v in m.items())+f" le5 {np.mean([r['reg']<=0.05 for r in rows]):.3f} zero {np.mean([r['reg']==0 for r in rows]):.3f}")
