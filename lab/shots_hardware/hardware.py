"""Hardware-aware compilation and routed-circuit noise study.

For each scenario and each variant (depth p, coupling-pruning threshold tau):
  1. QUBO = repo modelling rule (adaptive weighted-L2 projection), either full (28 couplings) or sparse
     (sparse_qubo.py): 'prune' removes couplings with |J_ij| < tau max|J|; 'refit' keeps the same support and
     re-projects all coefficients onto it; 'tree' uses the fixed zero-SWAP heavy-hex support. The QAOA (current final config: split gamma, per-qubit beta, Gibbs eta=2,
     ramp init, <=300 evaluations) is trained NOISELESSLY on the (pruned) QUBO with the repo trainer.
  2. The bound circuit is compiled for an IBM heavy-hex device: CouplingMap.from_heavy_hex(5) (57 qubits),
     basis {rz, sx, x, cx} (and {rz, sx, x, ecr} for counts), optimization_level=3, seed_transpiler in
     0..3; the compile with the fewest 2-qubit gates is kept (a compile-time choice that never looks at the
     output distribution). Reported: 2-qubit gate count, depth, 2-qubit depth. All-to-all compile for reference.
  3. Noise model on the ROUTED circuit (so SWAPs pay noise): 2-qubit depolarizing p=e2 after every cx,
     1-qubit depolarizing e2/10 after every sx/x (rz virtual, noiseless), independent readout flip e2 on each
     measured bit. Exact output distribution via Aer density matrix on the active physical qubits.
     Metrics: exact-distribution lift/enrichment (infinite shots) and 4096-shot sampled lift/enrichment
     (multinomial, seed 0) evaluated with the official cost table.

    python lab/shots_hardware/hardware.py            -> results/hardware.jsonl, results/hardware.md
    python lab/shots_hardware/hardware.py --quick    (R + 10 VAL scenarios)
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.transpiler import CouplingMap  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: E402

import common as C  # noqa: E402
import sparse_qubo as SQ  # noqa: E402

E2 = [0.0, 1e-3, 3e-3, 1e-2]
# (mode, tau, p): full QUBO; prune-and-refit at tau; plain pruning (as experiments/noise.py); zero-SWAP tree support
VARIANTS = ([('full', 0.0, p) for p in (1, 2, 3, 4)] + [('refit', 0.1, p) for p in (1, 2, 3, 4)]
            + [('refit', 0.25, p) for p in (1, 2, 3, 4)] + [('tree', 0.0, p) for p in (1, 2, 3, 4)]
            + [('prune', t, p) for t in (0.1, 0.25) for p in (2, 4)])
HEX = CouplingMap.from_heavy_hex(5)
SEEDS = range(4)
ECR_SCENARIOS = 11      # ECR-basis counts only on R + 10 scenarios (they track the cx counts 1:1 +- routing noise)
SIM = AerSimulator(method='density_matrix')
OUT = C.HERE / 'results'


def train_variant(s, mode, tau, p):
    cfg = C.run.DEFAULTS
    off, Q = SQ.sparse_qubo(s, mode, tau)
    h, J, _ = C.qb.qubo_to_ising(Q, off)
    e = C.qb.energies(Q, off)
    a, x, hist = C.qaoa.train(h, J, e, p=p, cost_mode=cfg['cost_mode'], mixer_mode=cfg['mixer_mode'],
                              kind=cfg['kind'], eta=cfg['eta'], stage1_evals=cfg['stage1_evals'],
                              ramp_dts=cfg['ramp_dts'])
    assert len(hist) <= C.MAX_EVALS
    return a, x, Q, e


def compile_best(qc, basis, coupling=HEX, seeds=SEEDS):
    best = None
    for sd in seeds:
        tq = transpile(qc, coupling_map=coupling, basis_gates=basis, optimization_level=3, seed_transpiler=sd)
        n2 = sum(1 for ins in tq.data if ins.operation.num_qubits == 2 and ins.operation.name != 'barrier')
        key = (n2, tq.depth())
        if best is None or key < best[0]:
            best = (key, tq)
    return best[1]


def stats_of(tq):
    two = lambda ins: ins.operation.num_qubits == 2 and ins.operation.name != 'barrier'  # noqa: E731
    return {'n2q': sum(1 for ins in tq.data if two(ins)), 'depth': tq.depth(),
            'depth2q': tq.depth(two), 'n1q': sum(1 for ins in tq.data if ins.operation.num_qubits == 1
                                                 and ins.operation.name not in ('measure', 'barrier'))}


def compact(tq):
    """Restrict a routed circuit to its active physical qubits; return (circuit, qubits ordered by clbit)."""
    act = sorted({tq.find_bit(q).index for ins in tq.data if ins.operation.name != 'barrier' for q in ins.qubits})
    pos = {p: i for i, p in enumerate(act)}
    meas = {}
    out = QuantumCircuit(len(act))
    for ins in tq.data:
        name = ins.operation.name
        if name == 'barrier':
            continue
        qs = [pos[tq.find_bit(q).index] for q in ins.qubits]
        if name == 'measure':
            meas[tq.find_bit(ins.clbits[0]).index] = qs[0]
            continue
        out.append(ins.operation, qs)
    order = [meas[c] for c in range(8)]
    return out, order


def readout(probs, r):
    t = probs.reshape((2,) * 8)
    for ax in range(8):
        t0, t1 = np.take(t, 0, axis=ax), np.take(t, 1, axis=ax)
        t = np.stack([(1 - r) * t0 + r * t1, r * t0 + (1 - r) * t1], axis=ax)
    return t.reshape(256)


def noisy_probs(cq_, order, e2, gate2='cx'):
    c = cq_.copy()
    c.save_probabilities(qubits=order)   # index = sum_c bit_c 2^c = int(schedule string, 2)
    if e2 == 0:
        res = SIM.run(c).result()
    else:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(e2 / 10, 1), ['sx', 'x'])
        nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), [gate2])
        res = SIM.run(c, noise_model=nm).result()
    p = np.clip(np.asarray(res.data()['probabilities'], float), 0, None)
    p = p / p.sum()
    return readout(p, e2) if e2 > 0 else p


def job(args):
    si, s, mode, tau, p = args
    t0 = time.time()
    costs = C.tm.cost_table(s)
    a, x, Q, e = train_variant(s, mode, tau, p)
    qmin = int(np.argmin(e))
    qc = C.cq.build_circuit(a, x)
    ideal = a.probabilities(x)
    a2a = stats_of(transpile(qc, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=3, seed_transpiler=0))
    tq = compile_best(qc, ['rz', 'sx', 'x', 'cx'])
    tq_ecr = compile_best(qc, ['rz', 'sx', 'x', 'ecr'], seeds=range(2)) if si < ECR_SCENARIOS else None
    cc, order = compact(tq)
    rec = {'si': si, 'scenario': s.get('name'), 'mode': mode, 'p': p, 'tau': tau,
           'n_couplings': int((np.triu(Q, 1) != 0).sum()),
           'regret_rel': float((costs[qmin] - costs.min()) / costs.min()) if costs.min() > 0 else 0.0,
           'all_to_all': a2a, 'heavy_hex_cx': stats_of(tq), 'heavy_hex_ecr': stats_of(tq_ecr) if tq_ecr is not None else None,
           'active_qubits': cc.num_qubits, 'noise': {}}
    rng = np.random.default_rng(0)
    for e2 in E2:
        pr = noisy_probs(cc, order, e2)
        if e2 == 0:
            rec['routing_check_maxdiff'] = float(np.abs(pr - ideal).max())
            assert rec['routing_check_maxdiff'] < 1e-4, rec['routing_check_maxdiff']
        dm = C.bl.distribution_metrics(costs, pr)
        cnt = rng.multinomial(C.SHOTS, pr)
        counts = {C.tm.BITSTRINGS[i]: int(n) for i, n in enumerate(cnt) if n}
        sm = C.bl.sample_metrics(costs, counts)
        rec['noise'][str(e2)] = {'lift': dm['lift'], 'enrichment': dm['enrichment'], 'p_optimal': dm['p_optimal'],
                                 'lift_4096': sm['lift'], 'enrichment_4096': sm['enrichment']}
    rec['sec'] = time.time() - t0
    return rec


def summarize(rows):
    lines = ['# Heavy-hex compilation and routed-circuit noise (exact density matrix)', '',
             f'Scenarios: {len({r["si"] for r in rows})} (R + VAL seed 555). Device: heavy-hex d=5 (57 qubits), '
             'opt level 3, best of 4 transpiler seeds by 2q count.', '',
             '| QUBO | p | couplings | regret<=5% | cx all-to-all | cx heavy-hex | ecr heavy-hex | depth hh | 2q depth hh |'
             ' lift e2=0 | lift 1e-3 | lift 3e-3 | lift 1e-2 | min lift 3e-3 | enr e2=0 | enr 1e-3 | enr 3e-3 | enr 1e-2 |'
             ' lift4096 3e-3 |',
             '|' + '---|' * 19]
    agg = {}
    for mode, tau, p in VARIANTS:
        rr = [r for r in rows if r['mode'] == mode and r['p'] == p and r['tau'] == tau]
        if not rr:
            continue
        m = lambda f: float(np.mean([f(r) for r in rr]))  # noqa: E731
        L = lambda e2: [r['noise'][str(e2)]['lift'] for r in rr]  # noqa: E731
        d = {'mode': mode, 'p': p, 'tau': tau, 'n': len(rr), 'couplings': m(lambda r: r['n_couplings']),
             'frac_regret_le_5pct': m(lambda r: r['regret_rel'] <= 0.05 + 1e-12),
             'cx_a2a': m(lambda r: r['all_to_all']['n2q']), 'cx_hh': m(lambda r: r['heavy_hex_cx']['n2q']),
             'ecr_hh': float(np.mean([r['heavy_hex_ecr']['n2q'] for r in rr if r['heavy_hex_ecr'] and r['si'] < ECR_SCENARIOS])), 'depth_hh': m(lambda r: r['heavy_hex_cx']['depth']),
             'depth2q_hh': m(lambda r: r['heavy_hex_cx']['depth2q'])}
        for e2 in E2:
            d[f'lift_{e2}'] = float(np.mean(L(e2)))
            d[f'minlift_{e2}'] = float(np.min(L(e2)))
            d[f'enr_{e2}'] = m(lambda r: r['noise'][str(e2)]['enrichment'])
            d[f'lift4096_{e2}'] = m(lambda r: r['noise'][str(e2)]['lift_4096'])
            d[f'enr4096_{e2}'] = m(lambda r: r['noise'][str(e2)]['enrichment_4096'])
        agg[f'{mode}{tau}_p{p}'] = d
        lines.append(f"| {mode}{'' if mode in ('full', 'tree') else f' tau={tau}'} | {p} | {d['couplings']:.1f} | {d['frac_regret_le_5pct']:.2f} | {d['cx_a2a']:.0f} | "
                     f"{d['cx_hh']:.0f} | {d['ecr_hh']:.0f} | {d['depth_hh']:.0f} | {d['depth2q_hh']:.0f} | "
                     f"{d['lift_0.0']:.3f} | {d['lift_0.001']:.3f} | {d['lift_0.003']:.3f} | {d['lift_0.01']:.3f} | "
                     f"{d['minlift_0.003']:.3f} | {d['enr_0.0']:.2f} | {d['enr_0.001']:.2f} | {d['enr_0.003']:.2f} | "
                     f"{d['enr_0.01']:.2f} | {d['lift4096_0.003']:.3f} |")
    return '\n'.join(lines) + '\n', agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    scs = C.val_set()
    if a.quick:
        scs = scs[:11]
    tag = 'hardware_quick' if a.quick else 'hardware'
    path = OUT / f'{tag}.jsonl'
    done = set()
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows.append(r)
            done.add((r['si'], r['mode'], r['tau'], r['p']))
    jobs = [(i, s, mode, tau, p) for mode, tau, p in VARIANTS for i, s in enumerate(scs)
            if (i, mode, tau, p) not in done]
    with ProcessPoolExecutor(a.workers) as ex, open(path, 'a') as fh:
        for r in ex.map(job, jobs, chunksize=1):
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            rows.append(r)
            print(f"si={r['si']} {r['mode']} tau={r['tau']} p={r['p']} cx_hh={r['heavy_hex_cx']['n2q']} "
                  f"lift0={r['noise']['0.0']['lift']:.3f} lift3e-3={r['noise']['0.003']['lift']:.3f} {r['sec']:.1f}s",
                  flush=True)
    md, agg = summarize(rows)
    (OUT / f'{tag}.md').write_text(md)
    (OUT / f'{tag}_summary.json').write_text(json.dumps(agg, indent=1))
    print(md)


if __name__ == '__main__':
    main()
