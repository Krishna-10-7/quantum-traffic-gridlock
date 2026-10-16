"""Compile-only comparison: generic SABRE routing vs a QAOA swap network (line swap strategy) on heavy-hex.

The cost layer exp(-i gamma_J sum J_ij Z_i Z_j) consists of commuting 2-qubit terms, so it can be routed
with a fixed swap network (Weidenfeld et al. 2023, Qiskit Commuting2qGateRouter + SwapStrategy.from_line):
the 8 logical qubits sit on a line of 8 physical heavy-hex qubits, and odd/even layers of SWAPs bring every
coupled pair adjacent; RZZ+SWAP pairs are fused by the optimiser. The swap network is applied per cost
layer (route_layers tracks the logical permutation it leaves behind, so no un-swapping is needed).
Same trained parameters for both compiles (HW.train_variant, noiseless training); both routed circuits are
verified against the ideal distribution and then evaluated under the hardware.py noise model.

    python lab/shots_hardware/swapnet.py [--n 21] [--workers 2]   -> results/swapnet.json / .md
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from qiskit.transpiler import PassManager  # noqa: E402
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (  # noqa: E402
    Commuting2qGateRouter, FindCommutingPauliEvolutions, SwapStrategy)

import common as C  # noqa: E402
import hardware as HW  # noqa: E402
import sparse_qubo as SQ  # noqa: E402

N = 8


def evolution_circuit(a, x):
    """Bound QAOA circuit whose coupling layers are PauliEvolutionGates (same unitary as circuit.build_circuit)."""
    q = C.cq.qubit_of
    qc = QuantumCircuit(N, N)
    qc.h(range(N))
    for gf, gj, b in a.layer_angles(x):
        for i in range(N):
            if a.h[i] != 0:
                qc.rz(2 * gf[i] * a.h[i], q(i))
        terms = []
        for i in range(N):
            for j in range(i + 1, N):
                if a.J[i, j] != 0:
                    lab = ['I'] * N
                    lab[q(i)] = lab[q(j)] = 'Z'
                    terms.append((''.join(reversed(lab)), gj * a.J[i, j]))
        if terms:
            qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=1.0), range(N))
        for i in range(N):
            qc.rx(-2 * b[i], q(i))
    qc.barrier()
    for i in range(N):
        qc.measure(q(i), q(i))
    return qc


def find_path(cm, n=N):
    adj = {q: set() for q in range(cm.size())}
    for u, v in cm.get_edges():
        adj[u].add(v)
        adj[v].add(u)

    def dfs(path):
        if len(path) == n:
            return path
        for w in sorted(adj[path[-1]]):
            if w not in path:
                r = dfs(path + [w])
                if r:
                    return r
        return None
    for s in range(cm.size()):
        r = dfs([s])
        if r:
            return r
    raise RuntimeError('no path')


def route_layers(a, x):
    """Line-swap-network routing, one cost layer at a time.

    Commuting2qGateRouter tracks the qubit permutation between several commuting blocks incorrectly in
    qiskit 2.5 (KeyError on a non-line edge from the 2nd block on), so every cost layer is routed on its
    own from the trivial layout, after relabelling its ZZ terms to the CURRENT line positions; the
    permutation left by the SWAPs is tracked here (who[pos] = logical qubit at line position pos) and all
    single-qubit gates and the final measurements are placed on the current positions."""
    q = C.cq.qubit_of
    strat = SwapStrategy.from_line(list(range(N)))
    coloring = {(i, i + 1): i % 2 for i in range(N - 1)}
    coloring.update({(i + 1, i): i % 2 for i in range(N - 1)})
    who = list(range(N))                      # line position -> logical (Qiskit) qubit
    out = QuantumCircuit(N, N)
    out.h(range(N))
    for gf, gj, b in a.layer_angles(x):
        pos = {v: k for k, v in enumerate(who)}
        for i in range(N):
            if a.h[i] != 0:
                out.rz(2 * gf[i] * a.h[i], pos[q(i)])
        terms = []
        for i in range(N):
            for j in range(i + 1, N):
                if a.J[i, j] != 0:
                    lab = ['I'] * N
                    lab[pos[q(i)]] = lab[pos[q(j)]] = 'Z'
                    terms.append((''.join(reversed(lab)), gj * a.J[i, j]))
        if terms:
            blk = QuantumCircuit(N)
            blk.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=1.0), range(N))
            routed = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter(strat, coloring)]).run(blk)
            for ins in routed.data:
                qs = [routed.find_bit(t).index for t in ins.qubits]
                out.append(ins.operation, qs)
                if ins.operation.name == 'swap':
                    who[qs[0]], who[qs[1]] = who[qs[1]], who[qs[0]]
        pos = {v: k for k, v in enumerate(who)}
        for i in range(N):
            out.rx(-2 * b[i], pos[q(i)])
    out.barrier()
    pos = {v: k for k, v in enumerate(who)}
    for v in range(N):
        out.measure(pos[v], v)
    return out


def swapnet_compile(a, x, basis=('rz', 'sx', 'x', 'cx')):
    routed = route_layers(a, x)
    path = find_path(HW.HEX)
    best = None
    for sd in HW.SEEDS:
        tq = transpile(routed, coupling_map=HW.HEX, basis_gates=list(basis), optimization_level=3,
                       initial_layout=path, seed_transpiler=sd)
        key = (HW.stats_of(tq)['n2q'], tq.depth())
        if best is None or key < best[0]:
            best = (key, tq)
    return best[1]


def check(tq, ideal):
    cc, order = HW.compact(tq)
    pr = HW.noisy_probs(cc, order, 0.0)
    return float(np.abs(pr - ideal).max())


VARIANTS = [('full', 0.0, 2), ('full', 0.0, 3), ('full', 0.0, 4), ('refit', 0.1, 2), ('refit', 0.1, 4),
            ('refit', 0.25, 2), ('refit', 0.25, 3)]
E2S = (0.001, 0.003, 0.01)


def job(args):
    si, s, mode, tau, p = args
    costs = C.tm.cost_table(s)
    a, x, Q, e = HW.train_variant(s, mode, tau, p)
    ideal = a.probabilities(x)
    r = {'si': si, 'mode': mode, 'tau': tau, 'p': p, 'lift_ideal': C.bl.distribution_metrics(costs, ideal)['lift']}
    for name, tq in (('sabre', HW.compile_best(C.cq.build_circuit(a, x), ['rz', 'sx', 'x', 'cx'])),
                     ('swapnet', swapnet_compile(a, x))):
        cc, order = HW.compact(tq)
        r[name] = HW.stats_of(tq)
        r[name]['active_qubits'] = cc.num_qubits
        r['check_' + name] = float(np.abs(HW.noisy_probs(cc, order, 0.0) - ideal).max())
        for e2 in E2S:
            dm = C.bl.distribution_metrics(costs, HW.noisy_probs(cc, order, e2))
            r[name][f'lift_{e2}'] = dm['lift']
            r[name][f'enr_{e2}'] = dm['enrichment']
    assert r['check_swapnet'] < 1e-4 and r['check_sabre'] < 1e-4, r
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=21)
    ap.add_argument('--workers', type=int, default=2)
    a_ = ap.parse_args()
    scs = C.val_set()[:a_.n]
    jobs = [(si, s, mode, tau, p) for mode, tau, p in VARIANTS for si, s in enumerate(scs)]
    rows = []
    with ProcessPoolExecutor(a_.workers) as ex:
        for r in ex.map(job, jobs, chunksize=1):
            rows.append(r)
            print(r['mode'], r['tau'], r['p'], r['si'], r['sabre']['n2q'], r['swapnet']['n2q'],
                  f"{r['sabre']['lift_0.003']:.3f} {r['swapnet']['lift_0.003']:.3f}", flush=True)
    (C.HERE / 'results' / 'swapnet.json').write_text(json.dumps(rows, indent=1))
    lines = ['# SABRE vs QAOA swap-network routing on heavy-hex (cx basis, opt level 3, best of 4 transpiler seeds)', '',
             f'Scenarios: R + {a_.n - 1} VAL scenarios (seed 555); same trained parameters for both compiles; both '
             f'routed circuits verified against the ideal distribution (max deviation '
             f'{max(max(r["check_sabre"], r["check_swapnet"]) for r in rows):.1e}). Lift = exact noisy distribution, '
             'noise model of hardware.py.', '',
             '| QUBO | p | lift ideal | SABRE cx | swap-net cx | SABRE 2q depth | swap-net 2q depth | swap-net fewer cx in |'
             ' SABRE lift 1e-3 | swap-net lift 1e-3 | SABRE lift 3e-3 | swap-net lift 3e-3 | SABRE lift 1e-2 | swap-net lift 1e-2 |',
             '|' + '---|' * 14]
    for mode, tau, p in VARIANTS:
        rr = [r for r in rows if (r['mode'], r['tau'], r['p']) == (mode, tau, p)]
        f = lambda k, m: np.mean([r[k][m] for r in rr])  # noqa: E731
        lines.append(f"| {mode}{'' if mode == 'full' else f' tau={tau}'} | {p} | {np.mean([r['lift_ideal'] for r in rr]):.3f} | "
                     f"{f('sabre', 'n2q'):.0f} | {f('swapnet', 'n2q'):.0f} | {f('sabre', 'depth2q'):.0f} | "
                     f"{f('swapnet', 'depth2q'):.0f} | {np.mean([r['swapnet']['n2q'] < r['sabre']['n2q'] for r in rr]):.0%} | "
                     + ' | '.join(f"{f(k, f'lift_{e2}'):.3f}" for e2 in E2S for k in ('sabre', 'swapnet')) + ' |')
    md = '\n'.join(lines) + '\n'
    (C.HERE / 'results' / 'swapnet.md').write_text(md)
    print(md)


if __name__ == '__main__':
    main()
