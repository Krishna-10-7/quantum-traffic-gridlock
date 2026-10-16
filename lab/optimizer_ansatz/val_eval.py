"""Full-pipeline evaluation on VAL = [scenario R] + random_scenarios(60, seed=555).

Both arms go through the UNMODIFIED run.solve() (QUBO -> train -> Qiskit circuit -> circuit/trainer
consistency assert -> 4096 shots with StatevectorSampler(seed) -> official-scorer metrics).
The candidate arm only swaps the trainer: qaoa.train is replaced, inside this process, by an
adapter around lab trainers (nothing in the repo is modified).

    python lab/optimizer_ansatz/val_eval.py                 # current vs candidate on VAL, seed 0
    python lab/optimizer_ansatz/val_eval.py --robust        # + scenario R, seeds 0..4
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (thread env vars before numpy)
import numpy as np  # noqa: E402

import qaoa  # noqa: E402
import run  # noqa: E402
import trainers  # noqa: E402

ORIG_TRAIN = qaoa.train
OUT = common.HERE / 'results'
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- candidate definitions
# pipeline cfg goes to run.solve (ansatz shape); 'train' goes to the lab trainer.
CANDIDATES = {
    # multi-angle (per-field) cost + per-qubit mixer, p=4, NFT/Rotosolve optimiser
    'field_nft_p4': {'pipeline': {'p': 4, 'cost_mode': 'field', 'mixer_mode': 'qubit'},
                     'train': {'stage2': 'nft', 'gen_evals': 8, 'stage1_evals': 40}},
    'field_nft_p1': {'pipeline': {'p': 1, 'cost_mode': 'field', 'mixer_mode': 'qubit'},
                     'train': {'stage2': 'nft', 'gen_evals': 8, 'stage1_evals': 15}},
    # current ansatz (split cost, per-qubit mixer), only the optimiser changed
    'split_nft_p4': {'pipeline': {'p': 4, 'cost_mode': 'split', 'mixer_mode': 'qubit'},
                     'train': {'stage2': 'nft', 'gen_evals': 16, 'gen_rho': 0.1, 'stage1_evals': 40}},
}


def make_adapter(train_cfg):
    def adapter(h, J, energies, p=4, cost_mode='split', mixer_mode='qubit', kind='gibbs', eta=2.0, alpha=0.25,
                max_evals=300, ramp_dts=(0.75, 1.0, 1.25), stage1_evals=40, rhobeg=0.15, init='ramp', init_seed=0):
        cfg = {'p': p, 'cost_mode': cost_mode, 'mixer_mode': mixer_mode, 'kind': kind, 'eta': eta,
               'max_evals': max_evals, 'ramp_dts': tuple(ramp_dts), **train_cfg}
        fn = trainers.train_grow if 'grow' in cfg else trainers.train
        a, x, b = fn(h, J, energies, cfg)
        assert b.n <= 300 and len(b.hist) == b.n
        return a, x, list(b.hist)
    return adapter


def job(args):
    arm, i, seed = args
    s, _ = common.val_set()[i]
    if arm == 'current':
        qaoa.train = ORIG_TRAIN
        res = run.solve(s, seed)
    else:
        cand = CANDIDATES[arm]
        qaoa.train = make_adapter(cand['train'])
        try:
            res = run.solve(s, seed, cand['pipeline'])
        finally:
            qaoa.train = ORIG_TRAIN
    m = res['metrics']
    q = m['baselines']['6_qaoa_raw_samples']
    return {'arm': arm, 'scenario': m['scenario'], 'seed': seed, 'lift': q['lift'], 'enrichment': q['enrichment'],
            'p_optimal': q['p_optimal'], 'E_Q': q['E_Q'], 'C_star': q['C_star'], 'C_rand': q['C_rand'],
            'regret_rel': m['surrogate']['regret_rel'], 'exact_lift': m['qaoa_exact_distribution']['lift'],
            'exact_enrichment': m['qaoa_exact_distribution']['enrichment'],
            'evals': m['training']['objective_evaluations'], 'n_params': m['circuit']['n_parameters'],
            'ops': m['circuit']['ops'], 'depth': m['circuit']['depth'],
            'consistency': m['circuit']['max_abs_prob_diff_circuit_vs_trainer'],
            'schedule_cost': res['schedule']['official_cost'], 'hybrid': m['baselines']['7_hybrid_labelled']['cost'],
            'seconds': m['runtime_seconds']['total']}


def summary(rows):
    L = np.array([r['lift'] for r in rows])
    En = np.array([r['enrichment'] for r in rows])
    R = [r for r in rows if r['scenario'] == 'R']
    return {'n': len(rows), 'R_lift': R[0]['lift'] if R else None, 'R_enrichment': R[0]['enrichment'] if R else None,
            'mean_lift': float(L.mean()), 'min_lift': float(L.min()), 'median_lift': float(np.median(L)),
            'mean_enrichment': float(En.mean()), 'min_enrichment': float(En.min()),
            'mean_p_optimal': float(np.mean([r['p_optimal'] for r in rows])),
            'frac_regret_le_5pct': float(np.mean([r['regret_rel'] <= 0.05 for r in rows])),
            'frac_schedule_optimal': float(np.mean([r['schedule_cost'] == r['C_star'] for r in rows])),
            'max_evals': int(max(r['evals'] for r in rows)),
            'max_consistency_err': float(max(r['consistency'] for r in rows)),
            'mean_seconds': float(np.mean([r['seconds'] for r in rows])),
            'max_seconds': float(np.max([r['seconds'] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['current', 'field_nft_p4'])
    ap.add_argument('--robust', action='store_true', help='also scenario R with seeds 0..4')
    ap.add_argument('--tag', default='val')
    a = ap.parse_args()
    n = len(common.val_set())
    out = {}
    with Pool(2) as pool:
        for arm in a.arms:
            t = time.time()
            rows = pool.map(job, [(arm, i, 0) for i in range(n)])
            out[arm] = {'summary': summary(rows), 'rows': rows, 'wall': time.time() - t}
            print(arm, json.dumps(out[arm]['summary']), flush=True)
            if a.robust:
                rr = pool.map(job, [(arm, 0, sd) for sd in range(5)])
                lifts = [r['lift'] for r in rr]
                out[arm]['R_seeds'] = {'lifts': lifts, 'enrichments': [r['enrichment'] for r in rr],
                                       'min_lift_clipped': float(np.clip(min(lifts), 0, 1)),
                                       'p_optimal': [r['p_optimal'] for r in rr]}
                print(arm, 'R seeds 0-4', json.dumps(out[arm]['R_seeds']), flush=True)
    f = OUT / f'{a.tag}.json'
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(out)
    f.write_text(json.dumps(old, indent=1))


if __name__ == '__main__':
    main()
