"""How often does the ramp fallback of init='transfer+ramp' win the probe round? (DEV and VAL; QUBO objective only)

    python lab/transfer_init/fallback_count.py
Also prints the zero-shot (1-evaluation, untrained) exact lift of the transfer vector on each set.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import transfer_init as TI  # noqa: E402

theta = np.array(TI.angles('split_qubit_eta2'))
for name, data in (('DEV (seed 7)', C.dev_set(60)), ('VAL (R + seed 555)', C.val_set())):
    wins, zs = [], []
    for pr in C.problems(data):
        base = C.qaoa.Ansatz(pr.h, pr.J, 4, 'single', 'single')
        fin = C.qaoa.Ansatz(pr.h, pr.J, 4, 'split', 'qubit')
        fb = C.qaoa.make_objective(base, pr.E, 'gibbs', 0.25, 2.0)
        ff = C.qaoa.make_objective(fin, pr.E, 'gibbs', 0.25, 2.0)
        v_ramp = min(fb(base.ramp(dt)) for dt in (0.75, 1.0, 1.25))
        v_t = ff(theta)
        wins.append(v_ramp < v_t)
        zs.append(C.bl.distribution_metrics(pr.costs, fin.probabilities(theta))['lift'])
    print(f'{name}: ramp probe beats transfer probe in {int(np.sum(wins))}/{len(wins)} scenarios; '
          f'zero-shot transfer exact lift mean {np.mean(zs):.4f} min {np.min(zs):.3f}', flush=True)
