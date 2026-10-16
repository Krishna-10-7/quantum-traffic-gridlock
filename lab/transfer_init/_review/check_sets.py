"""Review: regenerate LEARN / DEV / VAL fresh and compare with the agent's cached pickles; check overlaps."""
import os, sys, pickle, json
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import traffic_model as tm

CACHE = ROOT / 'lab' / 'transfer_init' / 'cache'

def key(s):
    d = {k: v for k, v in s.items() if k != 'name'}
    return json.dumps(d, sort_keys=True)

fresh = {
    's31_n150.pkl': tm.random_scenarios(150, seed=31),
    's7_n60.pkl': tm.random_scenarios(60, seed=7),
    'val_R_s555_n60.pkl': [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(60, seed=555),
}
for f, sc in fresh.items():
    cached = pickle.loads((CACHE / f).read_bytes())
    same_s = len(cached) == len(sc) and all(a == b for (a, _), b in zip(cached, sc))
    dc = max(float(np.abs(np.asarray(c) - tm.cost_table(b)).max()) for (_, c), b in zip(cached, sc))
    print(f'{f}: n={len(cached)} scenarios identical to fresh: {same_s}; max |cached cost - fresh cost table| = {dc}')

val_keys = {key(s) for s in fresh['val_R_s555_n60.pkl']}
for f in ('s31_n150.pkl', 's7_n60.pkl'):
    ov = sum(key(s) in val_keys for s in fresh[f])
    print(f'{f}: scenarios identical to a VAL scenario: {ov}')
# R in learn/dev?
R = fresh['val_R_s555_n60.pkl'][0]
print('R name', R.get('name'))
