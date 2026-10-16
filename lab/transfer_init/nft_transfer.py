"""Does parameter-transfer initialisation help on top of the NFT/Rotosolve trainer of lab/optimizer_ansatz?

    python lab/transfer_init/nft_transfer.py dev            # selection on DEV = random_scenarios(60, seed=7), exact metrics
    python lab/transfer_init/nft_transfer.py val [labels]   # VAL, 4096 shots (seed 0), same pipeline as run.solve

The NFT trainer (lab/optimizer_ansatz/trainers.py, imported read-only) is used unchanged; the only thing that
differs between paired arms is the START of the multi-angle (cost_mode='field', mixer_mode='qubit') stage:

  ref   : 3 linear-ramp probes of standard QAOA + COBYLA on standard QAOA (stage1_evals), embed, NFT to 300
          (= the optimizer track's trainers.train / train_grow)
  xfer  : 1 evaluation at the stored transfer vector of the field/qubit ansatz of the same depth
          (transfer_angles.json, learned on random_scenarios(150, seed=31)), then the same NFT loop to 300
  grow  : identity-layer growth p=1 -> 4 (thresholds as the optimizer track); 'xfer' variant starts p=1 from the
          p=1 transfer vector instead of ramp probes + standard-QAOA COBYLA.
Every objective evaluation is counted by trainers.Budget (asserted <= 300). Objective = QUBO energies only.
"""
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (thread env vars before numpy)
import numpy as np  # noqa: E402

import transfer_init as TI  # noqa: E402

_spec = importlib.util.spec_from_file_location('oa_trainers', C.ROOT / 'lab' / 'optimizer_ansatz' / 'trainers.py')
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

qaoa = C.qaoa
WORKERS = int(os.environ.get('TI_WORKERS', '2'))
RES = C.HERE / 'results'
FIELD = {'stage2': 'nft', 'cost_mode': 'field', 'mixer_mode': 'qubit', 'gen_evals': 8}


def train_xfer(h, J, E, cfg, theta):
    """Transfer start + the optimizer track's NFT loop (trainers._nft_loop) on the field/qubit ansatz."""
    c = {**T.DEFAULT, **cfg}
    budget = T.Budget(c['max_evals'])
    a = qaoa.Ansatz(h, J, c['p'], c['cost_mode'], c['mixer_mode'])
    ff = budget.wrap(qaoa.make_objective(a, E, c['kind'], c['alpha'], c['eta']))
    x = np.asarray(theta, float).copy()
    assert x.size == a.n_params
    fx = ff(x)
    T._nft_loop(ff, x, fx, a, budget, c['kind'], c, c['max_evals'])
    return a, budget.best_x, budget


def train_grow_xfer(h, J, E, cfg, theta1):
    """trainers.train_grow with the p=1 start replaced by the p=1 transfer vector (1 evaluation)."""
    c = {**T.DEFAULT, **cfg}
    kind = c['kind']
    budget = T.Budget(c['max_evals'])
    thresholds = list(c['grow']) + [c['max_evals']]
    p = 1
    a = qaoa.Ansatz(h, J, 1, c['cost_mode'], c['mixer_mode'])
    ff = budget.wrap(qaoa.make_objective(a, E, kind, c['alpha'], c['eta']))
    x = np.asarray(theta1, float).copy()
    fx = ff(x)
    for stop in thresholds:
        x, fx = T._nft_loop(ff, x, fx, a, budget, kind, c, stop)
        if p == c['p'] or stop >= c['max_evals']:
            break
        g, b = a.split(x)
        a = qaoa.Ansatz(h, J, p + 1, c['cost_mode'], c['mixer_mode'])
        p += 1
        x = a.join(np.vstack([g, np.zeros((1, g.shape[1]))]), np.vstack([b, np.zeros((1, b.shape[1]))]))
        ff = budget.wrap(qaoa.make_objective(a, E, kind, c['alpha'], c['eta']))
        budget.best_v, budget.best_x = fx, x.copy()
    while p < c['p']:
        g, b = a.split(x)
        a = qaoa.Ansatz(h, J, p + 1, c['cost_mode'], c['mixer_mode'])
        p += 1
        x = a.join(np.vstack([g, np.zeros((1, g.shape[1]))]), np.vstack([b, np.zeros((1, b.shape[1]))]))
        budget.best_x = x.copy()
    return a, budget.best_x, budget


# label: (trainer, cfg, transfer entry or None)
ARMS = {
    'ref  field-nft p=4 (ramp + std stage1 40)': ('train', {**FIELD, 'p': 4, 'stage1_evals': 40}, None),
    'xfer field-nft p=4 (transfer start)': ('xfer', {**FIELD, 'p': 4}, 'field_qubit_eta2'),
    'ref  field-nft p=1 (ramp + std stage1 15)': ('train', {**FIELD, 'p': 1, 'stage1_evals': 15}, None),
    'xfer field-nft p=1 (transfer start)': ('xfer', {**FIELD, 'p': 1}, 'field_qubit_p1_eta2'),
    'ref  field-nft p=2 (ramp + std stage1 20)': ('train', {**FIELD, 'p': 2, 'stage1_evals': 20}, None),
    'xfer field-nft p=2 (transfer start)': ('xfer', {**FIELD, 'p': 2}, 'field_qubit_p2_eta2'),
    'ref  field-nft grow 1->4 @150/200/250': ('train_grow', {**FIELD, 'p': 4, 'stage1_evals': 15,
                                                            'grow': (150, 200, 250)}, None),
    'xfer field-nft grow 1->4 @150/200/250': ('grow_xfer', {**FIELD, 'p': 4, 'grow': (150, 200, 250)},
                                              'field_qubit_p1_eta2'),
}


def train_arm(h, J, E, arm):
    kind, cfg, entry = ARMS[arm]
    cfg = {'kind': 'gibbs', 'eta': 2.0, 'max_evals': 300, **cfg}
    if kind == 'train':
        a, x, b = T.train(h, J, E, cfg)
    elif kind == 'train_grow':
        a, x, b = T.train_grow(h, J, E, cfg)
    elif kind == 'xfer':
        a, x, b = train_xfer(h, J, E, cfg, TI.angles(entry))
    else:
        a, x, b = train_grow_xfer(h, J, E, cfg, TI.angles(entry))
    assert b.n <= 300 and len(b.hist) == b.n
    return a, x, b


def solve(s, seed, arm, costs, sample):
    t0 = time.time()
    offset, Q = C.qb.build_qubo(s)
    h, J, _ = C.qb.qubo_to_ising(Q, offset)
    E = C.qb.energies(Q, offset)
    a, x, b = train_arm(h, J, E, arm)
    f_true = qaoa.make_objective(a, E, 'gibbs', 0.25, 2.0)(x)
    probs = a.probabilities(x)
    out = {'name': s.get('name'), 'evals': b.n, 'f_true': float(f_true),
           'exact': C.bl.distribution_metrics(costs, probs),
           'regret_rel': float((costs[int(np.argmin(E))] - costs.min()) / costs.min()) if costs.min() > 0 else 0.0}
    if sample:
        qc = C.cq.build_circuit(a, x)
        assert np.abs(C.cq.circuit_probabilities(qc) - probs).max() < 1e-8
        m = C.bl.sample_metrics(costs, C.cq.sample(qc, 4096, seed))
        out['sampled'] = {k: m[k] for k in ('E_Q', 'lift', 'enrichment', 'p_optimal')}
    out['seconds'] = time.time() - t0
    return out


def _job(args):
    return solve(*args)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    only = sys.argv[2:]
    arms = [a for a in ARMS if not only or any(o in a for o in only)]
    data = C.dev_set(60) if mode == 'dev' else C.val_set()
    which = 'exact' if mode == 'dev' else 'sampled'
    out = {}
    for arm in arms:
        t = time.time()
        jobs = [(s, 0, arm, np.asarray(c, float), mode != 'dev') for s, c in data]
        with ProcessPoolExecutor(WORKERS) as ex:
            rows = list(ex.map(_job, jobs))
        if mode == 'dev':
            sm = C.summarize(rows, 'exact')
            extra = ''
        else:
            sm = C.summarize(rows[1:], 'sampled')
            extra = f" | R lift {rows[0]['sampled']['lift']:.4f} enr {rows[0]['sampled']['enrichment']:.2f}"
        f = float(np.mean([r['f_true'] for r in rows]))
        print(f'{arm:45s} {C.fmt(sm)} f {f:.4f} evals<={max(r["evals"] for r in rows)}{extra} ({time.time() - t:.0f}s)',
              flush=True)
        out[arm] = {'summary': sm, 'mean_f': f, 'rows': rows}
    f = RES / f'nft_{mode}.json'
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(out)
    f.write_text(json.dumps(old, indent=1))


if __name__ == '__main__':
    main()
