"""Headline numbers of the optimizer_ansatz lab (reproduces RESULTS.md).

    python lab/optimizer_ansatz/final_eval.py            # all arms on VAL + scenario R seeds 0-4 + checks
    python lab/optimizer_ansatz/final_eval.py --quick    # current vs candidate only

VAL = [scenario R] + traffic_model.random_scenarios(60, seed=555). Every arm goes through the UNMODIFIED
run.solve() (QUBO -> train -> Qiskit circuit -> circuit/trainer consistency assert -> 4096 shots with
StatevectorSampler(seed) -> official scorer). Metric: 4096-shot SAMPLED lift / enrichment / P(opt), sampling
seed 0 (run.solve(s, 0)['metrics']['baselines']['6_qaoa_raw_samples']), identical for every arm.
Non-current arms only swap the trainer inside this process (qaoa.train -> lab trainer); no repo file changes.
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

import circuit as cq  # noqa: E402
import qaoa  # noqa: E402
import qaoa_nft  # noqa: E402
import qubo as qb  # noqa: E402
import run  # noqa: E402
import trainers  # noqa: E402

ORIG_TRAIN = qaoa.train
OUT = common.HERE / 'results'

# Frozen candidate (chosen on tuning seeds 31 and 7 only; see RESULTS.md): multi-angle QAOA with one angle per
# field term + one coupling angle per layer + per-qubit mixer, p = 4, trained by qaoa_nft.train
# (stage 1: 3 ramp probes + COBYLA on standard QAOA to 10 evals; stage 2: NFT sweeps + 16-eval COBYLA blocks
# on the coupling angles, to 300 evals).
CAND_PIPE = {'p': 4, 'cost_mode': 'field', 'mixer_mode': 'qubit', 'stage1_evals': 10}

ARMS = {
    'current': None,
    'candidate_field_nft_p4': ('nft', CAND_PIPE, {}),
    # ---- depth ablation (same trainer, same settings)
    'field_nft_p1': ('nft', {**CAND_PIPE, 'p': 1}, {}),
    'field_nft_p2': ('nft', {**CAND_PIPE, 'p': 2}, {}),
    'field_nft_p3': ('nft', {**CAND_PIPE, 'p': 3}, {}),
    # ---- entangling-layer ablation: J := 0 in the ansatz (objective still uses the full QUBO energies)
    'field_nft_p4_NO_couplings': ('nft', CAND_PIPE, {'no_coupling': True}),
    'field_nft_p1_NO_couplings': ('nft', {**CAND_PIPE, 'p': 1}, {'no_coupling': True}),
    # ---- optimiser alone / ansatz alone
    'field_ansatz_current_COBYLA_trainer': ('orig', {'cost_mode': 'field'}, {}),
    'split_ansatz_interp_cobyqa': ('lab', {}, {'layerwise': (10, 20, 30, 40), 'stage2': 'cobyqa', 'rho2': 0.075}),
    'split_ansatz_cobyqa_stage2': ('lab', {}, {'stage2': 'cobyqa', 'rho2': 0.075}),
}
R_SEED_ARMS = ('current', 'candidate_field_nft_p4', 'field_nft_p1', 'split_ansatz_interp_cobyqa')


def adapter(kind, extra):
    no_coup = extra.get('no_coupling', False)
    lab_cfg = {k: v for k, v in extra.items() if k != 'no_coupling'}

    def f(h, J, energies, **kw):
        if no_coup:
            J = np.zeros_like(J)
        if kind == 'nft':
            return qaoa_nft.train(h, J, energies, **kw)
        if kind == 'orig':
            return ORIG_TRAIN(h, J, energies, **kw)
        cfg = {'p': kw['p'], 'cost_mode': kw['cost_mode'], 'mixer_mode': kw['mixer_mode'], 'kind': kw['kind'],
               'eta': kw['eta'], 'max_evals': kw['max_evals'], 'ramp_dts': tuple(kw['ramp_dts']),
               'stage1_evals': kw['stage1_evals'], **lab_cfg}
        a, x, b = trainers.train(h, J, energies, cfg)
        assert b.n == len(b.hist)
        return a, x, list(b.hist)
    return f


def solve_arm(arm, s, seed):
    spec = ARMS[arm]
    if spec is None:
        qaoa.train = ORIG_TRAIN
        return run.solve(s, seed)
    kind, pipe, extra = spec
    qaoa.train = adapter(kind, extra)
    try:
        return run.solve(s, seed, pipe)
    finally:
        qaoa.train = ORIG_TRAIN


def state_stats(a, x):
    psi = a.statevector(x).reshape((2,) * 8)
    S = []
    for i in range(8):
        m = np.moveaxis(psi, i, 0).reshape(2, -1)
        ev = np.clip(np.linalg.eigvalsh(m @ m.conj().T).real, 1e-300, 1)
        S.append(float(-(ev * np.log2(ev)).sum()))
    pr = np.abs(psi.ravel()) ** 2
    H = lambda v: float(-(v[v > 0] * np.log2(v[v > 0])).sum())
    marg = [pr.reshape((2,) * 8).sum(axis=tuple(j for j in range(8) if j != i)) for i in range(8)]
    tc = sum(H(m) for m in marg) - H(pr)
    return float(np.mean(S)), float(tc)


def job(args):
    arm, i, seed = args
    s, _ = common.val_set()[i]
    t = time.time()
    res = solve_arm(arm, s, seed)
    wall = time.time() - t
    m = res['metrics']
    qr = m['baselines']['6_qaoa_raw_samples']
    spec = ARMS[arm]
    h = np.array(res['qubo']['ising']['h'])
    J = np.zeros((8, 8)) if (spec and spec[2].get('no_coupling')) else np.array(res['qubo']['ising']['J_upper'])
    cfg = m['config']
    a = qaoa.Ansatz(h, J, cfg['p'], cfg['cost_mode'], cfg['mixer_mode'])
    x = np.array(res['training']['params'])
    S1, tc = state_stats(a, x)
    gk, bk = a.split(x)
    ident = int(sum(np.all(gk[k] == 0) and np.all(bk[k] == 0) for k in range(a.p)))
    return {'arm': arm, 'scenario': m['scenario'], 'seed': seed, 'lift': qr['lift'], 'enrichment': qr['enrichment'],
            'p_optimal': qr['p_optimal'], 'E_Q': qr['E_Q'], 'C_star': qr['C_star'], 'C_rand': qr['C_rand'],
            'regret_rel': m['surrogate']['regret_rel'], 'exact_lift': m['qaoa_exact_distribution']['lift'],
            'exact_enrichment': m['qaoa_exact_distribution']['enrichment'],
            'evals': m['training']['objective_evaluations'], 'n_history': len(res['training']['history']),
            'n_params': m['circuit']['n_parameters'], 'ops': m['circuit']['ops'], 'depth': m['circuit']['depth'],
            'consistency': m['circuit']['max_abs_prob_diff_circuit_vs_trainer'],
            'schedule_cost': res['schedule']['official_cost'],
            'hybrid': m['baselines']['7_hybrid_labelled']['cost'],
            'train_seconds': m['runtime_seconds']['train'], 'seconds': m['runtime_seconds']['total'], 'wall': wall,
            'single_qubit_entropy_bits': S1, 'total_correlation_bits': tc, 'identity_layers': ident,
            'n_rzz': int(m['circuit']['ops'].get('rzz', 0))}


def summary(rows):
    L = np.array([r['lift'] for r in rows])
    En = np.array([r['enrichment'] for r in rows])
    R = [r for r in rows if r['scenario'] == 'R']
    return {'n': len(rows), 'R_lift': R[0]['lift'] if R else None, 'R_enrichment': R[0]['enrichment'] if R else None,
            'R_p_optimal': R[0]['p_optimal'] if R else None,
            'mean_lift': float(L.mean()), 'min_lift': float(L.min()), 'median_lift': float(np.median(L)),
            'mean_enrichment': float(En.mean()), 'min_enrichment': float(En.min()),
            'mean_p_optimal': float(np.mean([r['p_optimal'] for r in rows])),
            'mean_exact_lift': float(np.mean([r['exact_lift'] for r in rows])),
            'frac_regret_le_5pct': float(np.mean([r['regret_rel'] <= 0.05 for r in rows])),
            'frac_schedule_optimal': float(np.mean([r['schedule_cost'] == r['C_star'] for r in rows])),
            'max_evals': int(max(r['evals'] for r in rows)), 'max_history': int(max(r['n_history'] for r in rows)),
            'max_consistency_err': float(max(r['consistency'] for r in rows)),
            'mean_train_seconds': float(np.mean([r['train_seconds'] for r in rows])),
            'mean_seconds': float(np.mean([r['seconds'] for r in rows])),
            'max_seconds': float(np.max([r['seconds'] for r in rows])),
            'mean_single_qubit_entropy_bits': float(np.mean([r['single_qubit_entropy_bits'] for r in rows])),
            'mean_total_correlation_bits': float(np.mean([r['total_correlation_bits'] for r in rows])),
            'mean_identity_layers': float(np.mean([r['identity_layers'] for r in rows])),
            'n_params': rows[0]['n_params'], 'mean_rzz_gates': float(np.mean([r['n_rzz'] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--tag', default='final_val')
    a = ap.parse_args()
    arms = a.arms or (['current', 'candidate_field_nft_p4'] if a.quick else list(ARMS))
    n = len(common.val_set())
    f = OUT / f'{a.tag}.json'
    out = json.loads(f.read_text()) if f.exists() else {}
    with Pool(2) as pool:
        for arm in arms:
            t = time.time()
            rows = pool.map(job, [(arm, i, 0) for i in range(n)])
            out[arm] = {'summary': summary(rows), 'rows': rows, 'wall': time.time() - t}
            if arm in R_SEED_ARMS:
                rr = pool.map(job, [(arm, 0, sd) for sd in range(5)])
                lifts = [r['lift'] for r in rr]
                out[arm]['R_seeds'] = {'seeds': list(range(5)), 'lifts': lifts,
                                       'enrichments': [r['enrichment'] for r in rr],
                                       'p_optimal': [r['p_optimal'] for r in rr],
                                       'min_lift_clipped': float(np.clip(min(lifts), 0, 1)),
                                       'mean_lift': float(np.mean(lifts)),
                                       'seconds': [r['seconds'] for r in rr]}
            print(arm, json.dumps(out[arm]['summary']), flush=True)
            if 'R_seeds' in out[arm]:
                print(arm, 'R seeds 0-4', json.dumps(out[arm]['R_seeds']), flush=True)
            f.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
