"""Quick exploration: mean-objective transfer vectors (learned on LEARN, seed 31) evaluated zero-shot
(1 evaluation, no training) and with run-time COBYLA on DEV (seed 7), exact-distribution metrics."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

learn = C.problems(C.learn_set(150))
dev_data = C.dev_set(60)
dev = C.problems(dev_data)

for cm, mm in [('split', 'qubit'), ('single', 'single'), ('field', 'qubit')]:
    t = time.time()
    m = C.BatchQAOA(learn, 4, cm, mm, 'gibbs', 2.0)
    best = None
    for dt in (0.75, 1.0, 1.25):
        x, v = C.lbfgs(m, C.ramp(4, m.mg, m.mb, dt), maxiter=2000)
        if best is None or v < best[1]:
            best = (x, v)
    x, v = best
    md = C.BatchQAOA(dev, 4, cm, mm, 'gibbs', 2.0)
    pr = md.probabilities(x)
    rows = [{'exact': C.bl.distribution_metrics(p.costs, pr[i])} for i, p in enumerate(dev)]
    print(f'{cm}/{mm}: learn mean f {v:.4f}, dev mean f {md.value(x):.4f}; zero-shot DEV {C.fmt(C.summarize(rows, "exact"))}'
          f'  ({time.time() - t:.0f}s)', flush=True)
    np.save(C.CACHE / f'explore_bar_{cm}_{mm}.npy', x)
    print('   theta =', np.round(x, 3).tolist(), flush=True)
