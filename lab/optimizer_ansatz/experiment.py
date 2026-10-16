"""Tuning harness: run trainer configs on the tuning set (exact-distribution metrics, noise free).

    python lab/optimizer_ansatz/experiment.py <group> [--set tune40]

Results are appended to lab/optimizer_ansatz/results/tune.jsonl (one line per config).
"""
import argparse
import warnings
warnings.filterwarnings("ignore", message=".*MAXFUN.*")
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets thread env vars before numpy)
import numpy as np  # noqa: E402

import qaoa  # noqa: E402
import run  # noqa: E402
import trainers  # noqa: E402

OUT = common.HERE / 'results'
OUT.mkdir(exist_ok=True)
_SETS = {}


def get_set(key):
    if key not in _SETS:
        if key.startswith('tune'):
            n = int(key[4:].split('s')[0])
            seed = int(key.split('s')[1]) if 's' in key[4:] else 31
            _SETS[key] = common.tune_set(n, seed)
        else:
            raise ValueError(key)
    return _SETS[key]


def train_current(prob, cfg=None):
    d = {**run.DEFAULTS, **(cfg or {})}
    a, x, hist = qaoa.train(prob.h, prob.J, prob.E, p=d['p'], cost_mode=d['cost_mode'], mixer_mode=d['mixer_mode'],
                            kind=d['kind'], eta=d['eta'], max_evals=d['max_evals'], ramp_dts=d['ramp_dts'],
                            stage1_evals=d['stage1_evals'], init=d['init'], init_seed=0)
    return a, x, len(hist), float(min(hist))


def train_lab(prob, cfg):
    cfg = dict(cfg)
    J = prob.J * 0.0 if cfg.pop('no_coupling', False) else prob.J
    fn = trainers.train_grow if 'grow' in cfg else trainers.train
    a, x, b = fn(prob.h, J, prob.E, cfg)
    return a, x, b.n, float(b.best_v)


def job(args):
    cfg, key, i = args
    prob = get_set(key)[i]
    t = time.time()
    if cfg.get('trainer') == 'current':
        a, x, n, fbest = train_current(prob, {k: v for k, v in cfg.items() if k not in ('trainer', 'name')})
    else:
        a, x, n, fbest = train_lab(prob, {k: v for k, v in cfg.items() if k not in ('trainer', 'name')})
    assert n <= 300, n
    probs = a.probabilities(x)
    m = common.exact_metrics(prob, probs)
    # objective actually reached, recomputed independently (guards the analytic NFT bookkeeping)
    kind = cfg.get('kind', 'gibbs')
    f_true = qaoa.make_objective(a, prob.E, kind, 0.25, cfg.get('eta', 2.0))(x)
    m['n_params'] = a.n_params
    m.update(evals=n, f_best=fbest, f_true=float(f_true), seconds=time.time() - t, name=prob.name)
    return m


def evaluate(cfg, key='tune40', pool=None):
    jobs = [(cfg, key, i) for i in range(len(get_set(key)))]
    rows = pool.map(job, jobs) if pool else list(map(job, jobs))
    return rows


def report(name, cfg, key, rows):
    s = common.summarize(rows)
    drift = max(abs(r['f_best'] - r['f_true']) for r in rows)
    line = {'name': name, 'set': key, 'cfg': cfg, 'lift_mean': s['lift'][0], 'lift_min': s['lift'][1],
            'enr_mean': s['enrichment'][0], 'enr_min': s['enrichment'][1], 'popt_mean': s['p_optimal'][0],
            'max_evals_used': max(r['evals'] for r in rows), 'f_true_mean': float(np.mean([r['f_true'] for r in rows])),
            'bookkeeping_drift': drift, 'sec_mean': float(np.mean([r['seconds'] for r in rows])),
            'lifts': [r['lift'] for r in rows], 'enrs': [r['enrichment'] for r in rows]}
    with open(OUT / 'tune.jsonl', 'a') as fh:
        fh.write(json.dumps(line) + '\n')
    print(f"{name:42s} lift {line['lift_mean']:.4f} (min {line['lift_min']:.3f})  enr {line['enr_mean']:.2f} "
          f"(min {line['enr_min']:.2f})  P(opt) {line['popt_mean']:.3f}  f {line['f_true_mean']:.4f}  "
          f"evals<={line['max_evals_used']}  drift {drift:.1e}  {line['sec_mean']:.2f}s", flush=True)
    return line


GROUPS = {}


def group(name):
    def deco(fn):
        GROUPS[name] = fn
        return fn
    return deco


@group('smoke')
def g_smoke():
    return [('current', {'trainer': 'current'}),
            ('lab-cobyla (= current, memo)', {}),
            ('nft betas only', {'stage2': 'nft', 'gen_evals': 0}),
            ('nft + gen cobyla16', {'stage2': 'nft', 'gen_evals': 16})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('groups', nargs='+')
    ap.add_argument('--set', default='tune40')
    a = ap.parse_args()
    import experiment as _self
    import experiment_groups  # noqa: F401  (registers more groups into module 'experiment')
    GROUPS.update(_self.GROUPS)
    with Pool(2) as pool:
        for g in a.groups:
            for name, cfg in GROUPS[g]():
                rows = evaluate(cfg, a.set, pool)
                report(name, cfg, a.set, rows)


if __name__ == '__main__':
    main()
