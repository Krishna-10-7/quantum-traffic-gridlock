"""p-biased (mean-field) Taylor expansions: exact-expectation (P) and fluid (F) versions."""
import os, sys
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'): os.environ[_v]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parents[1])); sys.path.insert(0,str(HERE))
import itertools
import numpy as np, traffic_model as tm, baselines as bl, qubo as qb
from explore import components, switching_add, surrogate_metrics, summarize, fmt
from explore2 import local_tables, fit

def p_rule(s, rule):
    """P(NS green) per bit (8,), A1..A4,B1..B4."""
    a=s['arrivals']; f=s['link']['fraction']; lp=s['link']['phase']
    p=np.zeros(8)
    for n,node in enumerate('AB'):
        ns=np.array(a[node]['NS'],float); ew=np.array(a[node]['EW'],float)
        if node=='B' and rule.endswith('+link'):
            al=np.array(a['A'][lp],float); extra=np.r_[0, f*al[:3]]*0.5
            if lp=='NS': ns=ns+extra
            else: ew=ew+extra
        if rule.startswith('half'): pn=np.full(4,0.5)
        elif rule.startswith('flat'): pn=np.full(4, (ns.sum()+1e-9)/(ns.sum()+ew.sum()+2e-9))
        elif rule.startswith('cum'): pn=(np.cumsum(ns)+0.5)/(np.cumsum(ns)+np.cumsum(ew)+1)
        elif rule.startswith('slot'): pn=(ns+0.5)/(ns+ew+1)
        p[4*n:4*n+4]=pn
    return np.clip(p,0.1,0.9)

def build_pbiased(s, tabs, p):
    Q=np.zeros((8,8)); off=0.0
    for name,(vars_,pats,vals) in tabs.items():
        pv=p[vars_]; w=np.prod(np.where(pats==1,pv,1-pv),axis=1)
        off+=fit(vars_,pats,vals,w,Q)
    o=[off]; switching_add(s,Q,o); return o[0],Q

# ---------------- fluid version
def fluid_phi(arr, green, c, S, w, extra=(0,0,0,0)):
    q=0.0; cost=0.0; deps=[]
    for t in range(4):
        avail=q+arr[t]+extra[t]; d=min(avail, c*green[t]); q=avail-d
        cost+=q+w*max(0.0,q-S); deps.append(d)
    return cost, deps

def fluid_components(s, rounding=False):
    c,S,w=s['service_capacity'],s['storage_capacity'],s['overflow_weight']
    lp,f=s['link']['phase'],s['link']['fraction']; up='NS' if lp=='EW' else 'EW'
    G=lambda ph,b:[v if ph=='NS' else 1-v for v in b]; a=s['arrivals']
    def blink(x):
        _,d=fluid_phi(a['A'][lp],G(lp,[x[0],x[1],x[2],0.5]),c,S,w)
        tau=[0]+[(tm.transfer(f,int(round(d[t]))) if rounding else f*d[t]) for t in range(3)]
        return fluid_phi(a['B'][lp],G(lp,[x[4],x[5],x[6],x[7]]),c,S,w,tau)[0]
    return [('A_NS',[0,1,2,3],lambda x: fluid_phi(a['A']['NS'],G('NS',[x[i] for i in range(4)]),c,S,w)[0]),
            ('A_EW',[0,1,2,3],lambda x: fluid_phi(a['A']['EW'],G('EW',[x[i] for i in range(4)]),c,S,w)[0]),
            ('B_unl',[4,5,6,7],lambda x: fluid_phi(a['B'][up],G(up,[x[i] for i in range(4,8)]),c,S,w)[0]),
            ('B_link',[0,1,2,4,5,6,7],blink)]

def build_fluid(s, p, rounding=False):
    Q=np.zeros((8,8)); off=0.0; nev=0
    for name,vars_,F in fluid_components(s, rounding):
        base={i:float(p[i]) for i in range(8)}
        def ev(**kw):
            x=dict(base); x.update({int(k[1:]):v for k,v in kw.items()}); return F(x)
        f0=ev(); nev+=1
        d1={}; d2={}
        for i in vars_:
            d1[i]=ev(**{f'v{i}':1.0})-ev(**{f'v{i}':0.0}); nev+=2
        for i,j in itertools.combinations(vars_,2):
            d2[(i,j)]=ev(**{f'v{i}':1.0,f'v{j}':1.0})-ev(**{f'v{i}':1.0,f'v{j}':0.0})-ev(**{f'v{i}':0.0,f'v{j}':1.0})+ev(**{f'v{i}':0.0,f'v{j}':0.0}); nev+=4
        # quadratic: f0 + sum d1 (x-p) + sum d2 (x_i-p_i)(x_j-p_j)
        off+=f0
        for i in vars_:
            Q[i,i]+=d1[i]; off-=d1[i]*p[i]
        for (i,j),v in d2.items():
            Q[i,j]+=v; Q[i,i]-=v*p[j]; Q[j,j]-=v*p[i]; off+=v*p[i]*p[j]
    o=[off]; switching_add(s,Q,o); return o[0],Q,nev

if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 60
    scs=tm.random_scenarios(n, seed=31)
    res={}
    for s in scs:
        costs=tm.cost_table(s); tabs=local_tables(s); cands={}
        off,Q=qb.build_qubo(s,'adaptive'); cands['proj_adaptive']=qb.energies(Q,off)
        for rule in ('half','flat','cum','slot','flat+link','cum+link'):
            p=p_rule(s,rule)
            off,Q=build_pbiased(s,tabs,p); cands['P_'+rule]=qb.energies(Q,off)
            off,Q,_=build_fluid(s,p); cands['F_'+rule]=qb.energies(Q,off)
        for k,E in cands.items(): res.setdefault(k,[]).append(surrogate_metrics(E,costs))
    for k,rows in res.items(): print(fmt(summarize(rows,k)))
