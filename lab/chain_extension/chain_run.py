"""Chain pipeline (A -> B -> C, 12 qubits): QUBO -> budgeted QAOA -> exactly 4,096 shots -> chain-scorer metrics.

    python lab/chain_extension/chain_run.py --scenario lab/chain_extension/scenarios/R3.json --seed 0 --out <dir>

Same frozen design as run.py (p=4, split gamma, per-qubit beta, Gibbs eta=2, adaptive kappa=1, 3 ramp probes,
stage-1 standard QAOA to evaluation 40, total <= 300 evaluations). Writes qubo.json, circuit.qasm, counts.json,
schedule.json, metrics.json, training.json. Bit order A1..A4 B1..B4 C1..C4, 1 = NS green.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import chain_baselines as cb  # noqa: E402
import chain_circuit as cc  # noqa: E402
import chain_model as cm  # noqa: E402
import chain_qaoa as cq  # noqa: E402
import chain_qubo as cqb  # noqa: E402
import chain_scorer as cs  # noqa: E402

SHOTS = 4096
MAX_EVALS = 300
DEFAULTS = dict(p=4, cost_mode='split', mixer_mode='qubit', kind='gibbs', eta=2.0,
                measure='adaptive', kappa=1.0, sparsity='dense', build='global',
                stage1_evals=40, ramp_dts=(0.75, 1.0, 1.25), init='ramp', max_evals=MAX_EVALS)


def build(s, cfg, costs=None):
    if cfg['build'] == 'local':
        assert cfg['measure'] == 'uniform', 'local (approach-wise) construction is exact for the uniform measure'
        return cqb.build_qubo_local_uniform(s)
    return cqb.build_qubo(s, measure=cfg['measure'], kappa=cfg['kappa'], sparsity=cfg['sparsity'], total=costs)


def solve_chain(s, seed, cfg=None, shots=SHOTS, verify_circuit=True):
    cfg = {**DEFAULTS, **(cfg or {})}
    t0 = time.time()
    # QUBO (modelling rule).  The global construction evaluates the recurrence on all 4096 inputs (no optimum used).
    model_costs = None if cfg['build'] == 'local' else cm.cost_table(s)
    offset, Q, qinfo = build(s, cfg, model_costs)
    h, J, const = cqb.qubo_to_ising(Q, offset)
    energies = cqb.energies(Q, offset)
    t_qubo = time.time() - t0

    ansatz, params, history = cq.train(
        h, J, energies, p=cfg['p'], cost_mode=cfg['cost_mode'], mixer_mode=cfg['mixer_mode'], kind=cfg['kind'],
        eta=cfg['eta'], max_evals=min(cfg['max_evals'], MAX_EVALS), ramp_dts=cfg['ramp_dts'],
        stage1_evals=cfg['stage1_evals'], init=cfg['init'], init_seed=seed)
    assert len(history) <= MAX_EVALS
    t_train = time.time() - t0 - t_qubo

    qc = cc.build_circuit(ansatz, params)
    probs_numpy = ansatz.probabilities(params)
    consistency = None
    if verify_circuit:
        consistency = float(np.abs(cc.circuit_probabilities(qc) - probs_numpy).max())
        assert consistency < 1e-8, f'circuit/statevector mismatch {consistency}'
    counts = cc.sample(qc, shots, seed)
    t_quantum = time.time() - t0 - t_qubo - t_train

    # raw decoding: lowest QUBO energy among samples (official cost not used)
    mode = max(counts, key=lambda b: (counts[b], b))
    best_e = min(counts, key=lambda b: (energies[int(b, 2)], b))

    # ---------- benchmarks (chain scorer) ----------
    costs = cm.cost_table(s)
    qubo_min = cm.BITSTRINGS[int(np.argmin(energies))]
    m = cb.sample_metrics(costs, counts)
    c_star = float(costs.min())
    report = {
        'scenario': s.get('name'), 'seed': seed,
        'config': {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
        'baselines': {
            '1_best_alternating': cb.alternating(s, costs),
            '2_greedy_larger_queue': cb.greedy(s),
            '3_exact_enumeration': cb.exact(costs),
            '4_uniform_random_expected_cost': float(costs.mean()),
            '5_qubo_exact_minimizer': {'bits': qubo_min, 'cost': float(costs[int(qubo_min, 2)])},
            '6_qaoa_raw_samples': m,
            '7_hybrid_labelled': cb.hybrid_local_search(counts, costs),
        },
        'decoded': {'bits': best_e, 'cost': float(costs[int(best_e, 2)]),
                    'most_frequent': {'bits': mode, 'shots': counts[mode], 'cost': float(costs[int(mode, 2)])}},
        'surrogate': {
            'regret_rel': float((costs[int(qubo_min, 2)] - c_star) / c_star) if c_star > 0 else 0.0,
            'pearson': float(np.corrcoef(energies, costs)[0, 1]),
            **qinfo,
        },
        'qaoa_exact_distribution': cb.distribution_metrics(costs, probs_numpy),
        'circuit': {'qubits': qc.num_qubits, 'depth': qc.depth(), 'ops': dict(qc.count_ops()), 'p': cfg['p'],
                    'n_parameters': ansatz.n_params, 'max_abs_prob_diff_circuit_vs_trainer': consistency},
        'training': {'objective_evaluations': len(history), 'final_objective': float(min(history))},
        'runtime_seconds': {'qubo': t_qubo, 'train': t_train, 'circuit_and_sampling': t_quantum,
                            'total_incl_benchmarks': time.time() - t0},
    }
    return {'qubo': {'offset': offset, 'Q_upper_triangular': Q.tolist(), 'ising': {'h': h.tolist(), 'J_upper': J.tolist(),
                                                                                   'const': const},
                     'variable_to_schedule_bit': {str(i): f"{'ABC'[i // 4]}_slot{i % 4 + 1}" for i in range(12)},
                     'variable_to_qubit': {str(i): 11 - i for i in range(12)}, 'auxiliary_variables': []},
            'qasm': cc.to_qasm3(qc), 'counts': counts, 'metrics': report,
            'training': {'history': history, 'params': params.tolist()}}


def summarize(res):
    r = res['metrics']
    b = r['baselines']
    q = b['6_qaoa_raw_samples']
    return {'scenario': r['scenario'], 'seed': r['seed'], 'C_star': q['C_star'], 'C_rand': q['C_rand'], 'E_Q': q['E_Q'],
            'lift': q['lift'], 'enrichment': q['enrichment'], 'p_optimal': q['p_optimal'],
            'alternating': b['1_best_alternating']['cost'], 'greedy': b['2_greedy_larger_queue']['cost'],
            'qubo_min': b['5_qubo_exact_minimizer']['cost'], 'decoded': r['decoded']['cost'],
            'best_sampled': q['best_sampled']['cost'], 'hybrid': b['7_hybrid_labelled']['cost'],
            'regret_rel': r['surrogate']['regret_rel'], 'pearson': r['surrogate']['pearson'],
            'nonzero_couplings': r['surrogate']['nonzero_couplings'], 'simulations': r['surrogate']['simulations'],
            'exact_lift': r['qaoa_exact_distribution']['lift'], 'exact_enrichment': r['qaoa_exact_distribution']['enrichment'],
            'evals': r['training']['objective_evaluations'], 'rzz': r['circuit']['ops'].get('rzz', 0),
            'depth': r['circuit']['depth'], 'seconds': r['runtime_seconds']['total_incl_benchmarks'],
            'seconds_pipeline': r['runtime_seconds']['qubo'] + r['runtime_seconds']['train']
            + r['runtime_seconds']['circuit_and_sampling']}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    s = json.loads(Path(a.scenario).read_text())
    cs.validate_chain(s)
    res = solve_chain(s, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'qubo.json').write_text(json.dumps(res['qubo'], indent=1))
    (out / 'circuit.qasm').write_text(res['qasm'])
    (out / 'counts.json').write_text(json.dumps(res['counts'], indent=1))
    (out / 'metrics.json').write_text(json.dumps(res['metrics'], indent=2))
    (out / 'training.json').write_text(json.dumps(res['training']))
    dec = res['metrics']['decoded']
    (out / 'schedule.json').write_text(json.dumps({
        'bit_order': 'A_slot1..4 B_slot1..4 C_slot1..4; 1 = NS green, 0 = EW green (Qiskit keys read in this order)',
        'schedule': dec['bits'], 'A': [int(c) for c in dec['bits'][:4]], 'B': [int(c) for c in dec['bits'][4:8]],
        'C': [int(c) for c in dec['bits'][8:]], 'chain_cost': dec['cost'],
        'selection_rule': 'raw QAOA: lowest QUBO energy among the 4096 sampled bitstrings (chain scorer not used)',
        'alternatives': {'qaoa_most_frequent': dec['most_frequent'],
                         'qubo_exact_minimizer': res['metrics']['baselines']['5_qubo_exact_minimizer'],
                         'hybrid_labelled': res['metrics']['baselines']['7_hybrid_labelled']}}, indent=2))
    d = summarize(res)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}))


if __name__ == '__main__':
    main()
