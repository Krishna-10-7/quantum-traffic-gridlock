"""What do the entangling (ZZ) layers add?  Reads results/final_val.json (from final_eval.py).

    python lab/optimizer_ansatz/analyze_coupling.py

Per VAL scenario compares the frozen candidate (field-NFT p=4) with the same trainer on the same
ansatz with the coupling layer removed (J := 0 in the circuit; the objective still uses the full QUBO),
and splits scenarios by a coupling-strength statistic computed from the QUBO alone:
  frustrated  <=> the field-only minimiser z_i = -sign(h_i) is NOT the QUBO ground state
  rho = sum_{i<j} |J_ij| / sum_i |h_i|
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import numpy as np  # noqa: E402

import qubo as qb  # noqa: E402
import run  # noqa: E402


def qubo_stats(s):
    off, Q = qb.build_qubo(s, measure=run.DEFAULTS['measure'], kappa=run.DEFAULTS['kappa'])
    h, J, _ = qb.qubo_to_ising(Q, off)
    E = qb.energies(Q, off)
    z_field = -np.sign(h)
    z_field[z_field == 0] = 1
    x_field = ((1 - z_field) / 2).astype(int)          # x = (1 - z)/2
    idx_field = int(''.join(map(str, x_field)), 2)
    return {'frustrated': bool(E[idx_field] > E.min() + 1e-9),
            'rho': float(np.abs(np.triu(J, 1)).sum() / max(np.abs(h).sum(), 1e-12))}


def main():
    d = json.loads((common.HERE / 'results' / 'final_val.json').read_text())
    val = common.val_set()
    st = [qubo_stats(s) for s, _ in val]
    out = {}
    for coup, nocoup in (('candidate_field_nft_p4', 'field_nft_p4_NO_couplings'),
                         ('field_nft_p1', 'field_nft_p1_NO_couplings')):
        if coup not in d or nocoup not in d:
            continue
        a, b = d[coup]['rows'], d[nocoup]['rows']
        dl = np.array([x['lift'] - y['lift'] for x, y in zip(a, b)])
        de = np.array([x['enrichment'] - y['enrichment'] for x, y in zip(a, b)])
        fr = np.array([t['frustrated'] for t in st])
        rho = np.array([t['rho'] for t in st])
        res = {'n': len(dl), 'mean_dlift': float(dl.mean()), 'mean_denrichment': float(de.mean()),
               'wins': int((dl > 1e-3).sum()), 'losses': int((dl < -1e-3).sum()), 'ties': int((abs(dl) <= 1e-3).sum()),
               'min_lift_coupled': float(min(x['lift'] for x in a)), 'min_lift_uncoupled': float(min(y['lift'] for y in b)),
               'n_frustrated': int(fr.sum()),
               'mean_dlift_frustrated': float(dl[fr].mean()) if fr.any() else None,
               'mean_dlift_unfrustrated': float(dl[~fr].mean()) if (~fr).any() else None,
               'corr_dlift_rho': float(np.corrcoef(dl, rho)[0, 1]) if dl.std() > 0 else None,
               'mean_single_qubit_entropy_coupled': d[coup]['summary']['mean_single_qubit_entropy_bits'],
               'mean_total_correlation_coupled': d[coup]['summary']['mean_total_correlation_bits'],
               'worst5_dlift': sorted([(float(x), r['scenario']) for x, r in zip(dl, a)])[:5],
               'best5_dlift': sorted([(float(x), r['scenario']) for x, r in zip(dl, a)])[-5:]}
        out[f'{coup} vs {nocoup}'] = res
        print(coup, 'vs', nocoup, json.dumps(res, indent=1))
    for arm in ('current', 'split_ansatz_interp_cobyqa'):
        if arm in d:
            print(arm, 'entropy', d[arm]['summary']['mean_single_qubit_entropy_bits'],
                  'TC', d[arm]['summary']['mean_total_correlation_bits'])
    (common.HERE / 'results' / 'coupling_analysis.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
