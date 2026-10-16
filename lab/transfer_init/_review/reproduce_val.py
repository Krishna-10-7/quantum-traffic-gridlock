"""Review: independent reproduction of the transfer_init VAL headline numbers.

Arms (all on VAL = [R] + random_scenarios(60, seed=555), 4096 shots, StatevectorSampler(seed)):
  current      run.solve(s, seed)                          (repo code, unmodified)
  cand         agent's common.solve_ti with init='transfer+ramp'
  xfer         agent's common.solve_ti with init='transfer'
  rand         agent's common.solve_ti with init='random' (ablation)
  mine         reviewer's own strict trainer: 1 eval at theta_bar, then scipy COBYLA (rhobeg 0.075) with
               NO memo (COBYLA's re-evaluation of x0 is counted), hard cap 300 calls.
Instrumentation: qaoa.make_objective is wrapped so that every call of the training objective (any arm) is
counted at the source, and the `energies` array handed to it is checked against the QUBO energies.
Writes only to lab/transfer_init/_review/.
"""
import os
import sys
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TI = ROOT / 'lab' / 'transfer_init'
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import minimize

import baselines as bl
import circuit as cq
import qaoa
import qubo as qb
import traffic_model as tm

THETA = np.array(json.loads((TI / 'transfer_angles.json').read_text())['split_qubit_eta2']['theta'])
CALLS = {'n': 0, 'bad_energies': 0, 'E': None}
_orig_make = qaoa.make_objective


def _patched_make(ansatz, energies, *a, **k):
    if CALLS['E'] is None or not np.array_equal(np.asarray(energies), CALLS['E']):
        CALLS['bad_energies'] += 1
    f = _orig_make(ansatz, energies, *a, **k)

    def g(params):
        CALLS['n'] += 1
        return f(params)
    return g


qaoa.make_objective = _patched_make


def my_train(h, J, E, max_evals=300):
    a = qaoa.Ansatz(h, J, 4, 'split', 'qubit')
    f = _orig_make(a, E, 'gibbs', 0.25, 2.0)
    st = {'n': 0, 'v': np.inf, 'x': None}

    class Stop(Exception):
        pass

    def g(x):
        if st['n'] >= max_evals:
            raise Stop
        st['n'] += 1
        CALLS['n'] += 1
        v = f(x)
        if v < st['v']:
            st.update(v=v, x=np.array(x, float))
        return v
    g(THETA)
    try:
        minimize(g, THETA.copy(), method='COBYLA', options={'maxiter': max_evals - 1, 'rhobeg': 0.075, 'tol': 1e-7})
    except Stop:
        pass
    return a, st['x'], st['n']


def job(args):
    arm, idx, s, seed = args
    sys.path.insert(0, str(TI))
    import run
    import common as C
    off, Q = qb.build_qubo(s)
    h, J, _ = qb.qubo_to_ising(Q, off)
    E = qb.energies(Q, off)
    costs = tm.cost_table(s)
    CALLS.update(n=0, bad_energies=0, E=E)
    t0 = time.time()
    if arm == 'current':
        m = run.solve(s, seed)['metrics']
        q = m['baselines']['6_qaoa_raw_samples']
        out = {k: q[k] for k in ('lift', 'enrichment', 'p_optimal')}
        out['hist'] = m['training']['objective_evaluations']
        out['regret_rel'] = m['surrogate']['regret_rel']
    elif arm == 'mine':
        a, x, n = my_train(h, J, E)
        qc = cq.build_circuit(a, x)
        assert np.abs(cq.circuit_probabilities(qc) - a.probabilities(x)).max() < 1e-8
        counts = cq.sample(qc, 4096, seed)
        assert sum(counts.values()) == 4096
        q = bl.sample_metrics(costs, counts)
        out = {k: q[k] for k in ('lift', 'enrichment', 'p_optimal')}
        out['hist'] = n
    else:
        cfg = {'cand': dict(init='transfer+ramp', transfer=THETA.tolist()),
               'xfer': dict(init='transfer', transfer=THETA.tolist()),
               'rand': dict(init='random', init_seed=seed, cache=False)}[arm]
        r = C.solve_ti(s, seed, cfg, costs=costs)
        out = {k: r['sampled'][k] for k in ('lift', 'enrichment', 'p_optimal')}
        out['hist'] = r['evals']
    out.update(arm=arm, idx=idx, seed=seed, raw_calls=CALLS['n'], bad_energies=CALLS['bad_energies'],
               sec=time.time() - t0)
    return out


def main():
    arms = sys.argv[1:] or ['current', 'cand', 'xfer', 'mine', 'rand']
    R = tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')
    val = [R] + tm.random_scenarios(60, seed=555)
    jobs = [(a, i, s, 0) for a in arms for i, s in enumerate(val)]
    jobs += [(a, 0, R, sd) for a in arms if a != 'rand' for sd in range(1, 5)]
    t = time.time()
    with ProcessPoolExecutor(2) as ex:
        rows = list(ex.map(job, jobs))
    (HERE / f"val_rows_{'_'.join(arms)}.json").write_text(json.dumps(rows, indent=0))
    print(f'{len(rows)} jobs in {time.time() - t:.0f}s')
    for a in arms:
        rr = [r for r in rows if r['arm'] == a and r['seed'] == 0]
        rnd = [r for r in rr if r['idx'] > 0]
        L = np.array([r['lift'] for r in rnd])
        En = np.array([r['enrichment'] for r in rnd])
        P = np.array([r['p_optimal'] for r in rnd])
        Rr = [r for r in rr if r['idx'] == 0][0]
        Rs = sorted([r for r in rows if r['arm'] == a and r['idx'] == 0], key=lambda r: r['seed'])
        print(f"{a:8s} R lift {Rr['lift']:.4f} enr {Rr['enrichment']:.2f} | 60: mean lift {L.mean():.4f} min {L.min():.4f} "
              f"p10 {np.percentile(L, 10):.3f} enr {En.mean():.2f} (min {En.min():.2f}) P(opt) {P.mean():.3f} | "
              f"max hist {max(r['hist'] for r in rr)} max raw calls {max(r['raw_calls'] for r in rows if r['arm'] == a)} "
              f"bad_energies {sum(r['bad_energies'] for r in rows if r['arm'] == a)} | R seeds lifts "
              f"{[round(r['lift'], 4) for r in Rs]} min {min(r['lift'] for r in Rs):.4f} "
              f"enr {[round(r['enrichment'], 2) for r in Rs]}")
    if 'current' in arms:
        cur = {r['idx']: r for r in rows if r['arm'] == 'current' and r['seed'] == 0}
        for a in arms:
            if a == 'current':
                continue
            d = np.array([[r['lift'] - cur[r['idx']]['lift']] for r in rows if r['arm'] == a and r['seed'] == 0 and r['idx'] > 0])
            print(f'paired {a} - current: mean {d.mean():+.4f}, wins {int((d > 0).sum())}/60, ties {int((d == 0).sum())}')
        print('regret<=5% fraction (61):', np.mean([cur[i]['regret_rel'] <= 0.05 for i in cur]))


if __name__ == '__main__':
    main()
