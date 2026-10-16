"""Out-of-family sanity check on scenarios/edge/*.json (reporting only): current run.solve vs candidate, seed 0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import run  # noqa: E402
import transfer_init as TI  # noqa: E402

for f in sorted((C.ROOT / 'scenarios' / 'edge').glob('*.json')):
    s = C.tm.load_scenario(f)
    q = run.solve(s, 0)['metrics']['baselines']['6_qaoa_raw_samples']
    r = C.solve_ti(s, 0, TI.resolve(TI.CANDIDATE_CFG))['sampled']
    t = C.solve_ti(s, 0, TI.resolve(TI.ABLATION_VARIANTS[2][1]))['sampled']
    print(f"{s['name']}: C*={q['C_star']:.2f} C_rand={q['C_rand']:.2f} | current lift {q['lift']:.4f} enr {q['enrichment']:.2f} "
          f"| candidate lift {r['lift']:.4f} enr {r['enrichment']:.2f} | transfer-only lift {t['lift']:.4f} "
          f"enr {t['enrichment']:.2f}", flush=True)
