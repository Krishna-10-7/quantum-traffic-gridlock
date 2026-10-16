"""Finishing pass: robustness of the trainer choice on two tuning seeds (31 and 7), then freeze.

    python lab/optimizer_ansatz/experiment_robust.py robust --set tune40      (seed 31)
    python lab/optimizer_ansatz/experiment_robust.py robust --set tune40s7    (seed 7)

Same harness as experiment.py (exact-distribution metrics, results appended to results/tune.jsonl),
plus the number of layers left exactly at the identity (all angles 0) after training.
"""
import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore", message=".*MAXFUN.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402,F401  (thread env vars before numpy)
import numpy as np  # noqa: E402

import experiment as X  # noqa: E402
import qaoa  # noqa: E402
import trainers  # noqa: E402
import trainers2  # noqa: E402

FIELD = {'stage2': 'nft', 'cost_mode': 'field', 'gen_evals': 8}


def configs(name):
    if name == 'robust':
        return [
            ('field-nft p=4 direct gen12', {**FIELD, 'gen_evals': 12}),
            ('field-nft p=4 direct gen16', {**FIELD, 'gen_evals': 16}),
            ('field-nft p=4 direct s1=25', {**FIELD, 'stage1_evals': 25}),
            ('field-nft p=4 direct s1=10', {**FIELD, 'stage1_evals': 10}),
            ('field-nft p=4 grow (150,200,250) [old, layer-1-first]', {**FIELD, 'p': 4, 'stage1_evals': 15,
                                                                        'grow': (150, 200, 250)}),
            ('field-nft p=4 grow2 new-first (60,130,210)', {**FIELD, 'stage1_evals': 15, 'grow2': (60, 130, 210)}),
            ('field-nft p=4 grow2 new-first (100,170,240)', {**FIELD, 'stage1_evals': 15, 'grow2': (100, 170, 240)}),
            ('field-nft p=4 grow2 new-first (150,200,250)', {**FIELD, 'stage1_evals': 15, 'grow2': (150, 200, 250)}),
            ('field-nft p=4 stage1 layerwise INTERP 10/20/30/40', {**FIELD, 'layerwise': (10, 20, 30, 40)}),
            ('field-nft p=4 (gen8, s1=40) [reference]', dict(FIELD)),
            ('field-nft p=3 (s1=25)', {**FIELD, 'p': 3, 'stage1_evals': 25}),
            ('split cobyqa rho2=0.075', {'stage2': 'cobyqa', 'rho2': 0.075}),
            ('split layerwise INTERP 10/20/30/40 + cobyla', {'layerwise': (10, 20, 30, 40)}),
            ('split layerwise INTERP 10/20/30/40 + cobyqa', {'layerwise': (10, 20, 30, 40), 'stage2': 'cobyqa',
                                                           'rho2': 0.075}),
        ]
    if name == 'confirm':   # last combination check before freezing (gen16 and s1=10 were each better on both seeds)
        return [('field-nft p=4 direct s1=10 gen16', {**FIELD, 'stage1_evals': 10, 'gen_evals': 16})]
    if name == 'nocoup':
        return [('field-nft p=4 NO couplings', {**FIELD, 'no_coupling': True}),
                ('field-nft p=1 NO couplings', {**FIELD, 'p': 1, 'stage1_evals': 15, 'no_coupling': True})]
    raise ValueError(name)


def job(args):
    cfg, key, i = args
    prob = X.get_set(key)[i]
    t = time.time()
    c = dict(cfg)
    J = prob.J * 0.0 if c.pop('no_coupling', False) else prob.J
    if 'grow2' in c:
        c['grow'] = c.pop('grow2')
        a, x, b = trainers2.train_grow2(prob.h, J, prob.E, c)
    elif 'grow' in c:
        a, x, b = trainers.train_grow(prob.h, J, prob.E, c)
    else:
        a, x, b = trainers.train(prob.h, J, prob.E, c)
    assert b.n <= 300 and len(b.hist) == b.n
    probs = a.probabilities(x)
    m = common.exact_metrics(prob, probs)
    f_true = qaoa.make_objective(a, prob.E, c.get('kind', 'gibbs'), 0.25, c.get('eta', 2.0))(x)
    m.update(evals=b.n, f_best=float(b.best_v), f_true=float(f_true), seconds=time.time() - t, name=prob.name,
             identity_layers=trainers2.untouched_layers(a, x), p=a.p)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('groups', nargs='+')
    ap.add_argument('--set', default='tune40')
    a = ap.parse_args()
    with Pool(2) as pool:
        for g in a.groups:
            for name, cfg in configs(g):
                rows = pool.map(job, [(cfg, a.set, i) for i in range(len(X.get_set(a.set)))])
                X.report(name, cfg, a.set, rows)
                idl = [r['identity_layers'] for r in rows]
                print(f"{'':42s} identity layers: mean {np.mean(idl):.2f}  max {max(idl)} (of p={rows[0]['p']})",
                      flush=True)


if __name__ == '__main__':
    main()
