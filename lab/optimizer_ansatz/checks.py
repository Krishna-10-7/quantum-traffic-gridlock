"""Rule / integrity checks for the frozen candidate (single process).

    python lab/optimizer_ansatz/checks.py

1. parity      qaoa_nft.train (drop-in) == trainers.train (tuning harness) for the frozen config
2. budget      during training, the number of statevector simulations actually executed (counted by
               wrapping qaoa.Ansatz.statevector) equals len(history) and is <= 300 (probes included)
3. circuit     circuit.build_circuit(ansatz, params) -> Qiskit Statevector probabilities equal the
               trainer's probabilities to < 1e-9
4. determinism two independent run.solve calls (same seed) give identical params, counts and metrics
5. runtime     wall-clock seconds per run.solve call (scenario R, seeds 0-4), current vs candidate
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import numpy as np  # noqa: E402

import circuit as cq  # noqa: E402
import qaoa  # noqa: E402
import qaoa_nft  # noqa: E402
import run  # noqa: E402
import trainers  # noqa: E402
import final_eval as FE  # noqa: E402

OUT = common.HERE / 'results'
CAND = FE.CAND_PIPE


def main():
    rep = {}
    # 1. parity with the tuning harness
    probs = common.tune_set(40, 31)[:6]
    s1 = run.DEFAULTS['stage1_evals'] if 'stage1_evals' not in CAND else CAND['stage1_evals']
    d = []
    for pr in probs:
        a1, x1, h1 = qaoa_nft.train(pr.h, pr.J, pr.E, p=4, cost_mode='field', mixer_mode='qubit', stage1_evals=s1)
        a2, x2, b2 = trainers.train(pr.h, pr.J, pr.E, {'stage2': 'nft', 'cost_mode': 'field', 'gen_evals': 16,
                                                        'stage1_evals': s1})
        d.append((float(np.abs(x1 - x2).max()), len(h1), b2.n, bool(np.array_equal(h1, b2.hist))))
    rep['parity'] = {'max_param_diff': max(t[0] for t in d), 'history_lengths': [(t[1], t[2]) for t in d],
                     'histories_identical': all(t[3] for t in d)}
    print('parity', rep['parity'], flush=True)

    # 2. budget: count real simulations
    calls = {'n': 0}
    orig_sv = qaoa.Ansatz.statevector

    def counting_sv(self, params):
        calls['n'] += 1
        return orig_sv(self, params)
    val = common.val_set()
    rows = []
    qaoa.Ansatz.statevector = counting_sv
    try:
        for s, _ in val:
            calls['n'] = 0
            a, x, hist = qaoa_nft.train(*_ising(s), p=CAND['p'], cost_mode=CAND['cost_mode'],
                                        mixer_mode=CAND['mixer_mode'], kind='gibbs', eta=2.0, max_evals=300,
                                        ramp_dts=run.DEFAULTS['ramp_dts'], stage1_evals=s1)
            n_sim = calls['n']
            # 3. circuit reproduces the trained state
            pc = cq.circuit_probabilities(cq.build_circuit(a, x))
            pt = orig_sv(a, x)
            pt = np.abs(pt) ** 2
            rows.append({'scenario': s.get('name'), 'simulations': n_sim, 'history': len(hist),
                         'circuit_vs_trainer': float(np.abs(pc - pt / pt.sum()).max())})
    finally:
        qaoa.Ansatz.statevector = orig_sv
    rep['budget'] = {'all_simulations_counted': all(r['simulations'] == r['history'] for r in rows),
                     'max_history': max(r['history'] for r in rows), 'min_history': min(r['history'] for r in rows)}
    rep['circuit'] = {'max_abs_prob_diff': max(r['circuit_vs_trainer'] for r in rows), 'n_scenarios': len(rows)}
    print('budget', rep['budget'], flush=True)
    print('circuit', rep['circuit'], flush=True)

    # 4. determinism + 5. runtime (scenario R)
    R = val[0][0]
    det = []
    for sd in range(5):
        t = time.time()
        r1 = FE.solve_arm('candidate_field_nft_p4', R, sd)
        t1 = time.time() - t
        r2 = FE.solve_arm('candidate_field_nft_p4', R, sd)
        t = time.time()
        r0 = FE.solve_arm('current', R, sd)
        t0 = time.time() - t
        same = (r1['counts'] == r2['counts'] and r1['training']['params'] == r2['training']['params']
                and r1['training']['history'] == r2['training']['history'])
        det.append({'seed': sd, 'identical': same, 'cand_seconds': t1, 'current_seconds': t0,
                    'cand_lift': r1['metrics']['baselines']['6_qaoa_raw_samples']['lift'],
                    'cand_train_seconds': r1['metrics']['runtime_seconds']['train'],
                    'current_train_seconds': r0['metrics']['runtime_seconds']['train']})
        print('seed', det[-1], flush=True)
    rep['determinism_R'] = det
    rep['runtime'] = {'candidate_mean_seconds_per_seed': float(np.mean([r['cand_seconds'] for r in det])),
                      'current_mean_seconds_per_seed': float(np.mean([r['current_seconds'] for r in det])),
                      'candidate_mean_train_seconds': float(np.mean([r['cand_train_seconds'] for r in det])),
                      'current_mean_train_seconds': float(np.mean([r['current_train_seconds'] for r in det]))}
    print('runtime', rep['runtime'], flush=True)
    (OUT / 'checks.json').write_text(json.dumps({'checks': rep, 'budget_rows': rows}, indent=1))


def _ising(s):
    import qubo as qb
    off, Q = qb.build_qubo(s, measure=run.DEFAULTS['measure'], kappa=run.DEFAULTS['kappa'])
    h, J, _ = qb.qubo_to_ising(Q, off)
    return h, J, qb.energies(Q, off)


if __name__ == '__main__':
    main()
