"""Figures for the report (reads outputs/ and results/; run after benchmark.py and noise.py).

    python experiments/figures.py [--scenario-dir outputs/R]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import qubo as qb  # noqa: E402
import traffic_model as tm  # noqa: E402

LABELS = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4']
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'figure.dpi': 150})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default=str(ROOT / 'scenarios' / 'scenario_R.json'))
    ap.add_argument('--scenario-dir', default=str(ROOT / 'outputs' / 'R'))
    a = ap.parse_args()
    s = tm.load_scenario(a.scenario)
    d = Path(a.scenario_dir)
    name = s.get('name', 'X')
    costs = tm.cost_table(s)
    q = json.loads((d / 'seed0' / 'qubo.json').read_text())
    Q = np.array(q['Q_upper_triangular'])
    e = qb.energies(Q, q['offset'])
    counts = json.loads((d / 'seed0' / 'counts.json').read_text())

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    # (a) QUBO matrix
    M = Q + np.triu(Q, 1).T
    v = np.abs(M).max()
    im = ax[0].imshow(M, cmap='RdBu_r', vmin=-v, vmax=v)
    ax[0].set_xticks(range(8), LABELS, rotation=90)
    ax[0].set_yticks(range(8), LABELS)
    ax[0].set_title(f'(a) QUBO Q, scenario {name}')
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    # (b) surrogate fidelity
    ax[1].scatter(costs, e, s=4, alpha=.6)
    lo, hi = costs.min(), costs.max()
    ax[1].plot([lo, hi], [lo, hi], 'k--', lw=.7)
    k = int(np.argmin(e))
    ax[1].scatter([costs[k]], [e[k]], c='r', s=18, label='QUBO argmin')
    ax[1].set_xlabel('official cost')
    ax[1].set_ylabel('QUBO energy')
    ax[1].set_title(f'(b) r = {np.corrcoef(costs, e)[0, 1]:.3f}')
    ax[1].legend(loc='upper left', fontsize=7)
    # (c) sampled distribution by cost rank
    order = sorted(range(256), key=lambda i: (costs[i], tm.BITSTRINGS[i]))
    freq = np.array([counts.get(tm.BITSTRINGS[i], 0) for i in order]) / 4096
    ax[2].bar(range(256), freq, width=1.0)
    ax[2].axhline(1 / 256, color='k', lw=.7, ls='--', label='uniform')
    ax[2].axvline(12.5, color='r', lw=.7, ls=':', label='elite-13')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('schedules ranked by official cost')
    ax[2].set_ylabel('shot fraction')
    ax[2].set_title('(c) QAOA raw samples (seed 0)')
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    (ROOT / 'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT / 'figures' / f'fig_{name}.png', bbox_inches='tight')

    noise = ROOT / 'results' / 'noise.json'
    gen = ROOT / 'results' / 'generalization.json'
    if noise.exists() and gen.exists():
        n = json.loads(noise.read_text())
        g = json.loads(gen.read_text())
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))
        x = np.array(n['E2'])
        xs = np.where(x == 0, 3e-4, x)
        for v in n['variants']:
            ax[0].plot(xs, v['mean_lift'], marker='o', ms=3, label=f"{v['name']} ({v['cx_count_mean']:.0f} cx)")
        ax[0].set_xscale('log')
        ax[0].set_xlabel('2q depolarizing rate e2 (leftmost = noiseless)')
        ax[0].set_ylabel('mean lift')
        ax[0].set_title('(d) noise robustness')
        ax[0].legend(fontsize=6)
        rows = g['rows']
        ax[1].scatter([r['lift'] for r in rows], [r['enrichment'] for r in rows], s=6, alpha=.7)
        ax[1].axhline(256 / 13, color='k', lw=.7, ls='--', label='max enrichment 19.7')
        ax[1].set_xlabel(f"lift ({len(rows)} unseen random scenarios)")
        ax[1].set_ylabel('elite-13 enrichment')
        ax[1].legend(fontsize=6, loc='lower right')
        ax[1].set_title(f"(e) generalization: mean lift {g['lift']['mean']:.3f}, enrich. {g['enrichment']['mean']:.1f}")
        fig.tight_layout()
        fig.savefig(ROOT / 'figures' / 'fig_noise_generalization.png', bbox_inches='tight')
    print('figures written to', ROOT / 'figures')


if __name__ == '__main__':
    main()
