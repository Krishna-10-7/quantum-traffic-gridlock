"""Offline learning of scenario-independent QAOA transfer angles (training scenarios only).

    python lab/transfer_init/learn_angles.py                  # all entries -> transfer_angles.json
    python lab/transfer_init/learn_angles.py split_qubit_eta2 # one entry
    python lab/transfer_init/learn_angles.py --concentration  # per-instance optima analysis (slow, ~15 min)
    python lab/transfer_init/learn_angles.py --verify split_qubit_eta2   # regenerate and compare, no write

Training set LEARN = traffic_model.random_scenarios(150, seed=31) (hidden-scenario ranges). Never R, H, VAL
(seed 555) or the repo's reporting seeds (2026, 99).

For an ansatz A = (p=4, cost_mode, mixer_mode) and Gibbs temperature eta, f_s(theta) is exactly the run-time
training objective of scenario s (qaoa.make_objective 'gibbs' on the std-normalised QUBO Hamiltonian):

    f_s(theta) = -log sum_x |<x|psi_s(theta)>|^2 exp(-eta (E_s(x) - min E_s) / std E_s)

  mean-objective transfer   theta_bar = argmin_theta (1/|LEARN|) sum_{s in LEARN} f_s(theta)

computed with L-BFGS on the exact adjoint gradient (common.BatchQAOA), multi-start from linear ramps
dt in {0.5, 0.75, 1.0, 1.25, 1.5} and from already-learned vectors of nested ansatze (embedded exactly);
the start with the lowest mean objective wins. --concentration additionally computes per-instance optima
theta_s (L-BFGS from the current pipeline's trained angles) and their coordinate-wise median.
Everything is deterministic (fixed scenario seed, deterministic optimisers, fixed start list).
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

OUT = C.HERE / 'transfer_angles.json'
N_LEARN = 150
P = 4
RAMPS = (0.5, 0.75, 1.0, 1.25, 1.5)
# name: (cost_mode, mixer_mode, eta, extra starts = [(entry, (src cost_mode, src mixer_mode))])
SPECS = {
    'split_qubit_eta2': ('split', 'qubit', 2.0, []),
    'single_single_eta2': ('single', 'single', 2.0, []),
    'split_qubit_eta1': ('split', 'qubit', 1.0, [('split_qubit_eta2', ('split', 'qubit'))]),
    'split_qubit_eta3': ('split', 'qubit', 3.0, [('split_qubit_eta2', ('split', 'qubit'))]),
    'split_qubit_eta4': ('split', 'qubit', 4.0, [('split_qubit_eta2', ('split', 'qubit')),
                                                 ('split_qubit_eta3', ('split', 'qubit'))]),
    'field_qubit_eta2': ('field', 'qubit', 2.0, [('split_qubit_eta2', ('split', 'qubit'))]),
    'field_qubit_eta3': ('field', 'qubit', 3.0, [('split_qubit_eta3', ('split', 'qubit')),
                                                 ('field_qubit_eta2', ('field', 'qubit'))]),
    # shallower multi-angle ansatze, used as starts of the NFT/Rotosolve trainer of lab/optimizer_ansatz
    # (5th element = depth p; default P=4)
    'field_qubit_p1_eta2': ('field', 'qubit', 2.0, [], 1),
    'field_qubit_p2_eta2': ('field', 'qubit', 2.0, [], 2),
}


def learn(name, data, out):
    cm, mm, eta, extra, *rest = SPECS[name]
    p = rest[0] if rest else P
    t0 = time.time()
    model = C.BatchQAOA(C.problems(data), p, cm, mm, 'gibbs', eta)
    starts = [(f'ramp{dt}', C.ramp(p, model.mg, model.mb, dt)) for dt in RAMPS]
    for src, src_modes in extra:
        if src in out:
            starts.append((src, C.lift_vec(out[src]['theta'], p, src_modes, (cm, mm))))
    best, log = None, {}
    for sname, x0 in starts:
        x, v = C.lbfgs(model, x0, maxiter=3000)
        log[sname] = v
        print(f'    [{name}] start {sname:18s} -> mean objective {v:.5f}', flush=True)
        if best is None or v < best[2] - 1e-9:
            best = (sname, x, v)
    sname, x, v = best
    per = model.per_scenario(x)
    return {'name': name, 'p': p, 'cost_mode': cm, 'mixer_mode': mm, 'kind': 'gibbs', 'eta': eta,
            'n_learn': len(data), 'learn_seed': 31, 'theta': [float(round(t, 6)) for t in x],
            'winning_start': sname, 'mean_objective': v, 'objective_by_start': log,
            'per_scenario_objective_quartiles': np.percentile(per, [0, 25, 50, 75, 100]).tolist(),
            'seconds': time.time() - t0}


# ------------------------------------------------------------------ concentration analysis (optional)
def _instance_job(args):
    s, costs, cost_mode, mixer_mode, eta = args
    pr = C.Problem(s, costs)
    _, x_cur, _ = C.train_ti(pr.h, pr.J, pr.E, p=P, cost_mode=cost_mode, mixer_mode=mixer_mode, eta=eta, cache=False)
    model = C.BatchQAOA([pr], P, cost_mode, mixer_mode, 'gibbs', eta)
    x_opt, v_opt = C.lbfgs(model, x_cur, maxiter=400)
    return x_cur.tolist(), x_opt.tolist(), model.value(x_cur), v_opt


def concentration(data, out, name='split_qubit_eta2', workers=int(os.environ.get('TI_WORKERS', '1'))):
    cm, mm, eta = SPECS[name][:3]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_instance_job, [(s, c, cm, mm, eta) for s, c in data]))
    x_cur, x_opt = np.array([r[0] for r in res]), np.array([r[1] for r in res])
    v_cur, v_opt = np.array([r[2] for r in res]), np.array([r[3] for r in res])
    model = C.BatchQAOA(C.problems(data), P, cm, mm, 'gibbs', eta)
    theta_bar = np.array(out[name]['theta'])
    dist = lambda X: np.linalg.norm(X - theta_bar, axis=1)
    return {
        'ansatz': name,
        'mean_f_current_pipeline_300_evals': float(v_cur.mean()),
        'mean_f_per_instance_optimum': float(v_opt.mean()),
        'mean_f_at_median_of_per_instance_optima': model.value(np.median(x_opt, 0)),
        'mean_f_at_median_of_current_pipeline_angles': model.value(np.median(x_cur, 0)),
        'mean_f_at_theta_bar': model.value(theta_bar),
        'per_coordinate_iqr_of_optima': (np.percentile(x_opt, 75, 0) - np.percentile(x_opt, 25, 0)).tolist(),
        'median_distance_to_theta_bar': {'per_instance_optima': float(np.median(dist(x_opt))),
                                         'current_pipeline_angles': float(np.median(dist(x_cur)))},
    }


def main():
    args = sys.argv[1:]
    data = C.learn_set(N_LEARN)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    if '--concentration' in args:
        out.setdefault('_analysis', {})['concentration'] = concentration(data, out)
        OUT.write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out['_analysis']['concentration'].items() if 'iqr' not in k}, indent=1))
        return
    names = [a for a in args if a in SPECS] or list(SPECS)
    if '--verify' in args:   # recompute without writing; compare with the stored constants
        for name in names:
            new = learn(name, data, out)
            d = float(np.max(np.abs(np.array(new['theta']) - np.array(out[name]['theta']))))
            print(f'  [{name}] regenerated vs stored theta: max |diff| = {d:.2e} '
                  f"(mean objective {new['mean_objective']:.6f} vs {out[name]['mean_objective']:.6f})", flush=True)
        return
    for name in names:
        print(f'learning {name}', flush=True)
        out[name] = learn(name, data, out)
        OUT.write_text(json.dumps(out, indent=1))
        print(f"  [{name}] mean objective {out[name]['mean_objective']:.5f} from {out[name]['winning_start']} "
              f"({out[name]['seconds']:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
