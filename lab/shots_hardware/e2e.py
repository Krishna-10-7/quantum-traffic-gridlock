"""End-to-end emulated hardware run: shot-based training ON a noisy, heavy-hex-routed device.

Device emulation (precise definition):
  * the QAOA circuit is built with symbolic angles (ParameterVector), compiled ONCE for the heavy-hex
    coupling map (CouplingMap.from_heavy_hex(5)), basis {rz, sx, x, cx}, optimization_level 3, best of
    seed_transpiler 0..3 by 2-qubit count; parameters are bound per evaluation (no re-compilation),
    as in parametric execution on IBM hardware;
  * noise: 2-qubit depolarizing e2 after each cx, 1-qubit depolarizing e2/10 after each sx/x,
    readout bit flip e2; exact distribution from Aer density matrix on the active qubits;
  * every training evaluation = N shots sampled (seeded multinomial) from that noisy distribution;
    SPSA trainer of common.py, <= 300 evaluations; final metrics from 4096 noisy shots (seed 0) and
    from the exact noisy distribution.
Compared arms on the same routed noisy circuit:
  A  exact noiseless training (repo trainer)           -> noisy execution
  C  SPSA, N shots, noiseless device                  -> noisy execution
  B  SPSA, N shots, ON the noisy device (noise-aware)  -> noisy execution

    python lab/shots_hardware/e2e.py [--n 20] [--mode refit --tau 0.1 --p 2] [--e2 0.003] [--shots 1024]
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit import ParameterVector  # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: E402

import common as C  # noqa: E402
import hardware as HW  # noqa: E402
import sparse_qubo as SQ  # noqa: E402
SPSA = {'method': 'spsa', 'spsa_a': 0.6, 'spsa_c': 0.1, 'spsa_A': 10, 'spsa_avg': 0.25, 'select': 'last'}  # = shots_val.SPSA


def param_circuit(a):
    """Same gate sequence as circuit.build_circuit, with symbolic parameters (final-ansatz order)."""
    P = ParameterVector('t', a.n_params)
    p, mg, mb, n = a.p, a.mg, a.mb, C.qb.N
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    q = C.cq.qubit_of
    for k in range(p):
        g = [P[k * mg + j] for j in range(mg)]
        b = [P[p * mg + k * mb + j] for j in range(mb)]
        if a.cost_mode == 'single':
            gf, gj = [g[0]] * n, g[0]
        elif a.cost_mode == 'split':
            gf, gj = [g[0]] * n, g[1]
        else:
            gf, gj = g[:n], g[n]
        for i in range(n):
            if a.h[i] != 0:
                qc.rz(2 * float(a.h[i] / a.scale) * gf[i], q(i))
        for i in range(n):
            for j in range(i + 1, n):
                if a.J[i, j] != 0:
                    qc.rzz(2 * float(a.J[i, j] / a.scale) * gj, q(i), q(j))
        for i in range(n):
            qc.rx(-2 * (b[i] if mb > 1 else b[0]), q(i))
    qc.barrier()
    for i in range(n):
        qc.measure(q(i), q(i))
    return qc, P


class NoisyDevice:
    def __init__(self, a, e2):
        qc, self.P = param_circuit(a)
        best = None
        for sd in HW.SEEDS:
            tq = transpile(qc, coupling_map=HW.HEX, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=3,
                           seed_transpiler=sd)
            key = (HW.stats_of(tq)['n2q'], tq.depth())
            if best is None or key < best[0]:
                best = (key, tq)
        self.tq = best[1]
        self.stats = HW.stats_of(self.tq)
        cc, order = HW.compact(self.tq)
        cc.save_probabilities(qubits=order)
        self.cc, self.e2 = cc, e2
        self.nm = None
        if e2 > 0:
            self.nm = NoiseModel()
            self.nm.add_all_qubit_quantum_error(depolarizing_error(e2 / 10, 1), ['sx', 'x'])
            self.nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ['cx'])
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        bound = self.cc.assign_parameters({self.P[i]: float(v) for i, v in enumerate(x)}, inplace=False)
        res = HW.SIM.run(bound, noise_model=self.nm).result() if self.nm else HW.SIM.run(bound).result()
        pr = np.clip(np.asarray(res.data()['probabilities'], float), 0, None)
        pr /= pr.sum()
        return HW.readout(pr, self.e2) if self.e2 > 0 else pr


def metrics(costs, pr, seed=0):
    dm = C.bl.distribution_metrics(costs, pr)
    cnt = np.random.default_rng(seed).multinomial(C.SHOTS, pr)
    sm = C.bl.sample_metrics(costs, {C.tm.BITSTRINGS[i]: int(n) for i, n in enumerate(cnt) if n})
    return {'lift': dm['lift'], 'enrichment': dm['enrichment'], 'p_optimal': dm['p_optimal'],
            'lift_4096': sm['lift'], 'enrichment_4096': sm['enrichment']}


def job(args):
    si, s, mode, tau, p, e2, N, seed = args
    t0 = time.time()
    cfg = C.run.DEFAULTS
    costs = C.tm.cost_table(s)
    off, Q = SQ.sparse_qubo(s, mode, tau)
    h, J, _ = C.qb.qubo_to_ising(Q, off)
    e = C.qb.energies(Q, off)
    kw = dict(p=p, cost_mode=cfg['cost_mode'], mixer_mode=cfg['mixer_mode'], kind=cfg['kind'], eta=cfg['eta'],
              stage1_evals=cfg['stage1_evals'], ramp_dts=cfg['ramp_dts'])
    a, xA, hA = C.qaoa.train(h, J, e, **kw)
    dev = NoisyDevice(a, e2)
    ideal = dev.__class__.__new__(dev.__class__)   # same routed circuit, no noise (check)
    ideal.__dict__.update(dev.__dict__)
    ideal.nm, ideal.e2 = None, 0.0
    chk = float(np.abs(ideal(xA) - a.probabilities(xA)).max())
    assert chk < 1e-6, chk
    out = {'si': si, 'mode': mode, 'tau': tau, 'p': p, 'e2': e2, 'N': N, 'seed': seed, 'routed': dev.stats,
           'param_vs_ideal_check': chk}
    out['A_exact_noiseless_training'] = metrics(costs, dev(xA), seed)
    _, xC, hC = C.train_shots(h, J, e, N, seed, **kw, **{k: v for k, v in SPSA.items()})
    out['C_spsa_shots_noiseless_device'] = metrics(costs, dev(xC), seed)
    _, xB, hB = C.train_shots(h, J, e, N, seed, **kw, **{k: v for k, v in SPSA.items()}, prob_fn=dev)
    out['B_spsa_shots_on_noisy_device'] = metrics(costs, dev(xB), seed)
    out['evals'] = [len(hA), len(hC), len(hB)]
    assert max(out['evals']) <= C.MAX_EVALS
    out['sec'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20, help='R + first n-1 VAL scenarios')
    ap.add_argument('--mode', default='refit')
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--p', type=int, default=2)
    ap.add_argument('--e2', type=float, default=0.003)
    ap.add_argument('--shots', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    scs = C.val_set()[:a.n]
    tag = f'e2e_{a.mode}{a.tau}_p{a.p}_e{a.e2}_N{a.shots}'
    path = C.HERE / 'results' / f'{tag}.jsonl'
    done = {json.loads(line)['si'] for line in path.read_text().splitlines()} if path.exists() else set()
    jobs = [(i, s, a.mode, a.tau, a.p, a.e2, a.shots, 0) for i, s in enumerate(scs) if i not in done]
    with ProcessPoolExecutor(a.workers) as ex, open(path, 'a') as fh:
        for r in ex.map(job, jobs):
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            print(f"si={r['si']} cx={r['routed']['n2q']} A={r['A_exact_noiseless_training']['lift']:.3f} "
                  f"C={r['C_spsa_shots_noiseless_device']['lift']:.3f} B={r['B_spsa_shots_on_noisy_device']['lift']:.3f} "
                  f"{r['sec']:.0f}s", flush=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    summ = {}
    for arm in ('A_exact_noiseless_training', 'C_spsa_shots_noiseless_device', 'B_spsa_shots_on_noisy_device'):
        L = [r[arm]['lift'] for r in rows]
        L4 = [r[arm]['lift_4096'] for r in rows]
        En = [r[arm]['enrichment_4096'] for r in rows]
        summ[arm] = {'mean_lift': float(np.mean(L)), 'min_lift': float(np.min(L)), 'mean_lift_4096': float(np.mean(L4)),
                     'min_lift_4096': float(np.min(L4)), 'mean_enrichment_4096': float(np.mean(En)),
                     'R_lift_4096': next(r[arm]['lift_4096'] for r in rows if r['si'] == 0)}
        print(arm, json.dumps({k: round(v, 3) for k, v in summ[arm].items()}))
    (C.HERE / 'results' / f'{tag}_summary.json').write_text(json.dumps({'n': len(rows), 'cx_mean': float(np.mean(
        [r['routed']['n2q'] for r in rows])), 'arms': summ}, indent=1))


if __name__ == '__main__':
    main()
