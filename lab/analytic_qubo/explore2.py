"""Product-measure local projections (tuning set seed=31)."""
import os, sys
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'): os.environ[_v]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1])); sys.path.insert(0,str(HERE))
import itertools
import numpy as np, traffic_model as tm, baselines as bl, qubo as qb
from explore import components, switching_add, surrogate_metrics, summarize, fmt, gibbs_metrics

P4 = np.array(list(itertools.product([0,1], repeat=4)))

def local_tables(s):
    out = {}
    for name, vars_, F in components(s):
        k=len(vars_); pats=np.array(list(itertools.product([0,1], repeat=k)))
        vals=np.array([F({v:int(p[m]) for m,v in enumerate(vars_)}) for p in pats])
        out[name]=(vars_, pats, vals)
    return out

def fit(vars_, pats, vals, wts, Q):
    k=len(vars_); pairs=list(itertools.combinations(range(k),2))
    feats=np.column_stack([np.ones(len(pats))]+[pats[:,m] for m in range(k)]+[pats[:,a]*pats[:,b] for a,b in pairs])
    sw=np.sqrt(wts); coef,*_=np.linalg.lstsq(feats*sw[:,None], vals*sw, rcond=None)
    for m,v in enumerate(vars_): Q[v,v]+=coef[1+m]
    for n_,(a,b) in enumerate(pairs):
        i,j=vars_[a],vars_[b]; Q[min(i,j),max(i,j)]+=coef[1+k+n_]
    return coef[0]

def build_node_measure(s, tabs, wA, wB):
    """wA, wB: weights over 16 patterns of A bits / B bits (index = int of 4 bits)."""
    Q=np.zeros((8,8)); off=0.0
    for name,(vars_,pats,vals) in tabs.items():
        if name.startswith('A'):
            w = wA[pats@[8,4,2,1]]
        elif name=='B_unl':
            w = wB[pats@[8,4,2,1]]
        else:  # vars 0,1,2,4,5,6,7
            wA3 = wA.reshape(8,2).sum(1)  # marginal over A4
            w = wA3[pats[:,:3]@[4,2,1]] * wB[pats[:,3:]@[8,4,2,1]]
        off+=fit(vars_,pats,vals,w,Q)
    o=[off]; switching_add(s,Q,o); return o[0],Q

def prod_w(p4):
    return np.prod(np.where(P4==1, p4, 1-p4), axis=1)

def node_costs(s, tabs):
    """Local cost of node A plans (A_NS+A_EW+A switching) and node B plans (B_unl + B_link averaged over A uniform + B switching)."""
    sw = s['switch_weight']*np.array([sum(p[t]!=p[t-1] for t in range(1,4)) for p in P4])
    cA = tabs['A_NS'][2]+tabs['A_EW'][2]+sw
    cBu = tabs['B_unl'][2]
    _, pats, vals = tabs['B_link']
    return cA, cBu, pats, vals, sw

if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 60
    scs=tm.random_scenarios(n, seed=31)
    res={}
    for s in scs:
        costs=tm.cost_table(s); tabs=local_tables(s)
        cands={}
        off,Q=qb.build_qubo(s,'adaptive'); cands['proj_adaptive']=qb.energies(Q,off)
        off,Q=qb.build_qubo(s,'uniform'); cands['proj_uniform']=qb.energies(Q,off)
        u=np.ones(16)
        off,Q=build_node_measure(s,tabs,u,u); cands['local_uniform']=qb.energies(Q,off)
        g=bl.greedy(s)['bits']; gA=np.array([int(c) for c in g[:4]]); gB=np.array([int(c) for c in g[4:]])
        for lam in (0.2,0.4,0.6):
            off,Q=build_node_measure(s,tabs,prod_w(0.5+lam*(gA-0.5)),prod_w(0.5+lam*(gB-0.5)))
            cands[f'greedy_tilt{lam}']=qb.energies(Q,off)
        cA,cBu,pats,vals,sw=node_costs(s,tabs)
        # B link averaged over A uniform -> per B pattern
        bidx=pats[:,3:]@[8,4,2,1]
        cBl=np.array([vals[bidx==k].mean() for k in range(16)])
        cB=cBu+cBl+sw
        for kap in (0.5,1.0,2.0):
            wA=np.exp(-(cA-cA.min())/(kap*cA.std())); wB=np.exp(-(cB-cB.min())/(kap*cB.std()))
            off,Q=build_node_measure(s,tabs,wA,wB); cands[f'local_boltz{kap}']=qb.energies(Q,off)
        for k,E in cands.items(): res.setdefault(k,[]).append(surrogate_metrics(E,costs))
    for k,rows in res.items(): print(fmt(summarize(rows,k)))
