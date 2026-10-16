"""Run-time strategy selection on DEV = random_scenarios(n, seed=7) (never VAL / 2026 / 99).

    python lab/transfer_init/tune.py <experiment-name> [...]

Metric: exact-distribution (infinite-shot) lift / enrichment of the final trained circuit (noise-free selection);
the reporting script evaluate_val.py uses 4096 sampled shots (seed 0) exactly like run.solve.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

WORKERS = int(os.environ.get('TI_WORKERS', '1'))
RES = C.HERE / 'results'
RES.mkdir(exist_ok=True)


def vec(key):
    """Transfer vectors: 'json:<entry>:<field>' from transfer_angles.json or 'npy:<file>' from cache/."""
    kind, rest = key.split(':', 1)
    if kind == 'npy':
        return np.load(C.CACHE / rest).tolist()
    entry, field = rest.split(':')
    return json.loads((C.HERE / 'transfer_angles.json').read_text())[entry][field]


def resolve(cfg):
    cfg = dict(cfg)
    for k in ('transfer', 'transfer_base'):
        if isinstance(cfg.get(k), str):
            cfg[k] = vec(cfg[k])
    return cfg


def _job(args):
    s, costs, cfg, sample = args
    r = C.solve_ti(s, 0, cfg, costs=costs, sample=sample)
    r.pop('params')
    return r


def run_cfg(data, cfg, sample=False):
    cfg = resolve(cfg)
    jobs = [(s, c, cfg, sample) for s, c in data]
    if WORKERS == 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(_job, jobs))


BAR = 'npy:explore_bar_split_qubit.npy'
BAR_BASE = 'npy:explore_bar_single_single.npy'
BAR_FIELD = 'npy:explore_bar_field_qubit.npy'
J = lambda e: f'json:{e}:theta'

EXPERIMENTS = {
    'basic': [
        ('current (ramp, as run.solve)', dict(cache=False)),
        ('ramp, cached x0', dict()),
        ('transfer rich, rho2=0.075', dict(init='transfer', transfer=BAR)),
        ('transfer rich, rho2=0.03', dict(init='transfer', transfer=BAR, rhobeg2=0.03)),
        ('transfer rich, rho2=0.15', dict(init='transfer', transfer=BAR, rhobeg2=0.15)),
        ('transfer+ramp', dict(init='transfer+ramp', transfer=BAR)),
        ('transfer_base (std ansatz) + stage1/2', dict(init='transfer_base', transfer_base=BAR_BASE)),
        ('transfer field/qubit, rho2=0.075', dict(init='transfer', transfer=BAR_FIELD, cost_mode='field')),
    ],
    'round2': [
        ('transfer+ramp', dict(init='transfer+ramp', transfer=J('split_qubit_eta2'))),
        ('transfer_base (std ansatz) + stage1/2', dict(init='transfer_base', transfer_base=J('single_single_eta2'))),
        ('transfer + scale global3 (20)', dict(init='transfer', transfer=J('split_qubit_eta2'), scale_evals=20)),
        ('transfer + scale layer12 (40)', dict(init='transfer', transfer=J('split_qubit_eta2'), scale_evals=40,
                                               scale_mode='layer12')),
        ('transfer eta3', dict(init='transfer', transfer=J('split_qubit_eta3'), eta=3.0)),
        ('transfer eta4', dict(init='transfer', transfer=J('split_qubit_eta4'), eta=4.0)),
        ('transfer eta1', dict(init='transfer', transfer=J('split_qubit_eta1'), eta=1.0)),
        ('transfer field eta2', dict(init='transfer', transfer=J('field_qubit_eta2'), cost_mode='field')),
    ],
    # round2 was interrupted after the 'scale' entries; round3 = its remaining entries + ablation arms
    'round3': [
        ('transfer only (json split_qubit_eta2)', dict(init='transfer', transfer=J('split_qubit_eta2'))),
        ('transfer eta1', dict(init='transfer', transfer=J('split_qubit_eta1'), eta=1.0)),
        ('transfer eta3', dict(init='transfer', transfer=J('split_qubit_eta3'), eta=3.0)),
        ('transfer eta4', dict(init='transfer', transfer=J('split_qubit_eta4'), eta=4.0)),
        ('random init (3 probes, seed 0)', dict(init='random', init_seed=0, cache=False)),
        ('transfer field eta2', dict(init='transfer', transfer=J('field_qubit_eta2'), cost_mode='field')),
        ('transfer field eta3', dict(init='transfer', transfer=J('field_qubit_eta3'), cost_mode='field', eta=3.0)),
    ],
    # budget curve: how many evaluations does the transfer start save?
    'budget': [
        ('transfer, budget 1 (zero-shot, no training)', dict(init='transfer', transfer=J('split_qubit_eta2'), max_evals=1)),
        ('transfer, budget 50', dict(init='transfer', transfer=J('split_qubit_eta2'), max_evals=50)),
        ('transfer, budget 100', dict(init='transfer', transfer=J('split_qubit_eta2'), max_evals=100)),
        ('transfer, budget 200', dict(init='transfer', transfer=J('split_qubit_eta2'), max_evals=200)),
        ('ramp, budget 100', dict(init='ramp', cache=False, max_evals=100)),
        ('ramp, budget 200', dict(init='ramp', cache=False, max_evals=200)),
    ],
}


def main():
    names = sys.argv[1:] or ['basic']
    n = int(os.environ.get('TI_NDEV', '60'))
    data = C.dev_set(60)[:n]
    for name in names:
        out = []
        for label, cfg in EXPERIMENTS[name]:
            t = time.time()
            rows = run_cfg(data, cfg)
            sm = C.summarize(rows, 'exact')
            ev = float(np.mean([r['evals'] for r in rows]))
            fo = float(np.mean([r['final_objective'] for r in rows]))
            print(f'{label:45s} {C.fmt(sm)} f {fo:.4f} evals {ev:.0f} ({time.time() - t:.0f}s)', flush=True)
            out.append({'label': label, 'cfg': {k: v for k, v in cfg.items()}, 'summary': sm, 'mean_final_objective': fo,
                        'lifts': [r['exact']['lift'] for r in rows]})
        (RES / f'dev_{name}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
