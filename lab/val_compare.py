"""VAL comparison of the integrated pipeline: Webster vs adaptive QUBO, both with the NFT trainer."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import numpy as np
import benchmark as bm
import traffic_model as tm
if __name__ == '__main__':
    val = [tm.load_scenario(ROOT / 'scenarios' / 'scenario_R.json')] + tm.random_scenarios(60, seed=555)
    for name, cfg in [('webster+nft (new default)', {}), ('adaptive+nft', {'measure': 'adaptive'})]:
        rows = bm.pmap([(s, 0, cfg) for s in val])
        L = np.array([r['lift'] for r in rows]); E = np.array([r['enrichment'] for r in rows])
        P = np.array([r['p_optimal'] for r in rows]); reg = np.array([r['regret_rel'] for r in rows])
        print(f"{name:28s} R {L[0]:.4f}/{E[0]:.2f} | lift mean {L.mean():.4f} min {L.min():.4f} | enr mean {E.mean():.2f} min {E.min():.2f} "
              f"| P(opt) {P.mean():.3f} | regret<=5% {np.mean(reg <= .05):.3f} | evals max {max(r['evals'] for r in rows)} | sec max {max(r['seconds'] for r in rows):.1f}", flush=True)
