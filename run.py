"""Quantum Traffic Gridlock pipeline.

    python run.py --scenario scenarios/scenario_R.json --seed 0 --out outputs/R_seed0

Writes qubo.json, circuit.qasm, counts.json, schedule.json, metrics.json, training.json.
Bit order everywhere: A slot 1..4 then B slot 1..4; 1 = NS green, 0 = EW green.
CPU only, no network, deterministic for a given seed.
"""
import argparse
import json
import os
import time
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

import baselines as bl  # noqa: E402
import circuit as cq  # noqa: E402
import qaoa  # noqa: E402
import qaoa_nft  # noqa: E402
import qubo as qb  # noqa: E402
import traffic_model as tm  # noqa: E402

SHOTS = 4096
MAX_EVALS = 300
DEFAULTS = dict(p=4, cost_mode='field', mixer_mode='qubit', kind='gibbs', eta=2.0,
                measure='webster', kappa=1.0, stage1_evals=13, ramp_dts=(0.75, 1.0, 1.25),
                init='ramp', max_evals=MAX_EVALS, trainer='nft', gen_evals=16)


def train_qubo(offset, Q, cfg, seed):
    """Ising conversion + budgeted QAOA training for a given QUBO. Returns (ansatz, params, history, energies).

    cfg['no_couplings'] (ablation) drops the ZZ terms from the circuit's cost layer; the training
    objective still uses the full QUBO energies.
    """
    h, J, _ = qb.qubo_to_ising(Q, offset)
    energies = qb.energies(Q, offset)
    if cfg.get('no_couplings'):
        J = np.zeros_like(J)
    train_kw = dict(p=cfg['p'], cost_mode=cfg['cost_mode'], mixer_mode=cfg['mixer_mode'],
                    kind=cfg['kind'], eta=cfg['eta'], max_evals=min(cfg['max_evals'], MAX_EVALS),
                    ramp_dts=cfg['ramp_dts'], stage1_evals=cfg['stage1_evals'], init=cfg['init'], init_seed=seed)
    if cfg['trainer'] == 'nft':      # exact sequential minimal optimisation (NFT/Rotosolve), qaoa_nft.py
        ansatz, params, history = qaoa_nft.train(h, J, energies, gen_evals=cfg['gen_evals'], **train_kw)
    else:                            # COBYLA-only trainer (ablation; the previous submission)
        ansatz, params, history = qaoa.train(h, J, energies, **train_kw)
    assert len(history) <= MAX_EVALS
    return ansatz, params, history, energies


def solve(s, seed, cfg=None, shots=SHOTS):
    """Full pipeline on a scenario dict. Returns a dict with every artefact (no file I/O)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    t0 = time.time()
    offset, Q = qb.build_qubo(s, measure=cfg['measure'], kappa=cfg['kappa'])
    h, J, const = qb.qubo_to_ising(Q, offset)
    t_qubo = time.time() - t0
    ansatz, params, history, energies = train_qubo(offset, Q, cfg, seed)
    t_train = time.time() - t0 - t_qubo

    qc = cq.build_circuit(ansatz, params)
    probs_circuit = cq.circuit_probabilities(qc)
    probs_numpy = ansatz.probabilities(params)
    consistency = float(np.abs(probs_circuit - probs_numpy).max())
    assert consistency < 1e-8, f'circuit/statevector mismatch {consistency}'
    counts = cq.sample(qc, shots, seed)

    # ---------- decoding (raw QAOA, uses only the QUBO, never the official cost) ----------
    mode = max(counts, key=lambda b: (counts[b], b))
    best_e = min(counts, key=lambda b: (energies[int(b, 2)], b))

    # ---------- benchmarks (official scorer) ----------
    costs = tm.cost_table(s)
    qubo_min = tm.BITSTRINGS[int(np.argmin(energies))]
    m = bl.sample_metrics(costs, counts)
    report = {
        'scenario': s.get('name'),
        'seed': seed,
        'config': {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
        'baselines': {
            '1_best_alternating': bl.alternating(s),
            '2_greedy_larger_queue': bl.greedy(s),
            '3_exact_enumeration': bl.exact(costs),
            '4_uniform_random_expected_cost': float(costs.mean()),
            '5_qubo_exact_minimizer': {'bits': qubo_min, 'cost': float(costs[int(qubo_min, 2)]),
                                       'qubo_energy': float(energies.min())},
            '6_qaoa_raw_samples': m,
            '7_hybrid_labelled': bl.hybrid_local_search(s, counts, costs),
        },
        'surrogate': {
            'regret_abs': float(costs[int(qubo_min, 2)] - costs.min()),
            'regret_rel': float((costs[int(qubo_min, 2)] - costs.min()) / costs.min()) if costs.min() > 0 else 0.0,
            'pearson_qubo_vs_official': float(np.corrcoef(energies, costs)[0, 1]),
            'spearman_qubo_vs_official': float(np.corrcoef(np.argsort(np.argsort(energies)),
                                                           np.argsort(np.argsort(costs)))[0, 1]),
        },
        'qaoa_exact_distribution': bl.distribution_metrics(costs, probs_numpy),
        'circuit': {'qubits': qc.num_qubits, 'depth': qc.depth(), 'ops': dict(qc.count_ops()),
                    'p': cfg['p'], 'n_parameters': ansatz.n_params,
                    'max_abs_prob_diff_circuit_vs_trainer': consistency},
        'training': {'objective_evaluations': len(history), 'final_objective': float(min(history))},
        'runtime_seconds': {'qubo': t_qubo, 'train': t_train, 'total': time.time() - t0},
    }
    schedule = {
        'bit_order': 'A_slot1 A_slot2 A_slot3 A_slot4 B_slot1 B_slot2 B_slot3 B_slot4; 1 = NS green, 0 = EW green',
        'counts_key_order': 'identical to bit_order (schedule bit i is measured on qubit/clbit 7-i, so Qiskit keys read A1..B4)',
        'schedule': best_e,
        'selection_rule': 'raw QAOA: lowest QUBO energy among the 4096 sampled bitstrings (official scorer not used)',
        'A': [int(c) for c in best_e[:4]], 'B': [int(c) for c in best_e[4:]],
        'official_cost': float(costs[int(best_e, 2)]),
        'alternatives': {
            'qaoa_most_frequent': {'bits': mode, 'shots': counts[mode], 'official_cost': float(costs[int(mode, 2)])},
            'qubo_exact_minimizer': {'bits': qubo_min, 'official_cost': float(costs[int(qubo_min, 2)])},
            'hybrid_labelled': report['baselines']['7_hybrid_labelled'],
        },
    }
    qubo_json = {
        'description': ('E(x) = offset + sum_{i<=j} Q[i][j] x_i x_j, x in {0,1}^8. Per-approach second-order '
                        'expansion of the exact queue cost plus the exact switching term; see REPORT.md section 2.'),
        'n_variables': 8, 'auxiliary_variables': [],
        'variable_to_schedule_bit': {str(i): f"{'AB'[i // 4]}_slot{i % 4 + 1}" for i in range(8)},
        'variable_to_qubit': {str(i): cq.qubit_of(i) for i in range(8)},
        'bit_value_meaning': '1 = NS green, 0 = EW green',
        'offset': offset,
        'Q_upper_triangular': Q.tolist(),
        'ising': {'convention': 'x_i = (1 - z_i)/2; E = const + sum h_i z_i + sum_{i<j} J_ij z_i z_j',
                  'h': h.tolist(), 'J_upper': J.tolist(), 'const': const},
        'construction': {'measure': cfg['measure'],
                         'rule': ('second-order expansion of each approach exact queue cost at the Webster '
                                  'equal-saturation split (analytic_qubo.py); 176 single-approach recursions, '
                                  'no joint-schedule cost table' if cfg['measure'] == 'webster' else
                                  'weighted-L2 quadratic projection of the full cost table (qubo.py)')},
    }
    training = {'history': history, 'params': params.tolist(),
                'layer_angles': [{'field_gamma': gf.tolist(), 'coupling_gamma': float(gj), 'mixer_beta': b.tolist()}
                                 for gf, gj, b in ansatz.layer_angles(params)]}
    return {'qubo': qubo_json, 'qasm': cq.to_qasm3(qc), 'counts': counts, 'schedule': schedule,
            'metrics': report, 'training': training}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--p', type=int, default=DEFAULTS['p'], choices=[1, 2, 3, 4])
    ap.add_argument('--measure', default=DEFAULTS['measure'], choices=['webster', 'adaptive', 'uniform'])
    a = ap.parse_args()
    s = tm.load_scenario(a.scenario)
    res = solve(s, a.seed, {'p': a.p, 'measure': a.measure})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'qubo.json').write_text(json.dumps(res['qubo'], indent=2))
    (out / 'circuit.qasm').write_text(res['qasm'])
    (out / 'counts.json').write_text(json.dumps(res['counts'], indent=1))
    (out / 'schedule.json').write_text(json.dumps(res['schedule'], indent=2))
    (out / 'metrics.json').write_text(json.dumps(res['metrics'], indent=2))
    (out / 'training.json').write_text(json.dumps(res['training']))
    m = res['metrics']['baselines']
    q = m['6_qaoa_raw_samples']
    print(f"scenario={s.get('name')} seed={a.seed}  C*={q['C_star']}  C_rand={q['C_rand']:.3f}")
    print(f"QAOA raw: E_Q={q['E_Q']:.3f}  lift={q['lift']:.4f}  enrichment={q['enrichment']:.2f}  "
          f"P(opt)={q['p_optimal']:.3f}  schedule={res['schedule']['schedule']} "
          f"(cost {res['schedule']['official_cost']})")
    print(f"alternating={m['1_best_alternating']['cost']}  greedy={m['2_greedy_larger_queue']['cost']}  "
          f"QUBO-min={m['5_qubo_exact_minimizer']['cost']}  hybrid={m['7_hybrid_labelled']['cost']}  "
          f"evals={res['metrics']['training']['objective_evaluations']}  "
          f"time={res['metrics']['runtime_seconds']['total']:.1f}s  -> {out}")


if __name__ == '__main__':
    main()
