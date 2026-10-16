import os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; EXT = HERE.parent; ROOT = EXT.parents[1]
sys.path[:0] = [str(EXT), str(ROOT)]
import numpy as np
import chain_model as cm, chain_qubo as cqb, chain_qaoa as cq, chain_baselines as cb
old = json.loads((EXT / 'results' / 'diag_budget_seed31.json').read_text())['rows']
scen = cm.random_chain_scenarios(12, seed=31)
for k in (0, 5):
    s = scen[k]; costs = cm.cost_table(s)
    off, Q, _ = cqb.build_qubo(s); h, J, _ = cqb.qubo_to_ising(Q, off); e = cqb.energies(Q, off)
    for B in (300, 1000):
        a, x, hist = cq.train(h, J, e, p=4, cost_mode='split', mixer_mode='qubit', max_evals=B)
        print(s['name'], B, len(hist), round(cb.distribution_metrics(costs, a.probabilities(x))['lift'], 6), 'agent', round(old[k][f'final@{B}'], 6))
