"""Reproduce every headline number of the shots/hardware lab (from the repo root):

    python lab/shots_hardware/reproduce.py            # all stages (about 2.5 h on 2 cores)
    python lab/shots_hardware/reproduce.py shots hw   # selected stages

Stages (each script resumes from / overwrites its own files in lab/shots_hardware/results/):
  tune    tune_shots.py a..f   optimiser selection on random_scenarios(20, seed=31) only
  shots   shots_val.py         (a) shot-based training on VAL = [R] + random_scenarios(60, seed=555), R x 5 seeds
  hw      hardware.py          (b) heavy-hex compilation + routed-circuit depolarizing noise, 18 variants x 61 scenarios
  swapnet swapnet.py           (b) SABRE vs QAOA swap-network routing (R + 10 VAL scenarios)
  e2e     e2e.py               (a)+(b) SPSA shot training ON the noisy routed device (R + 19 VAL scenarios)
  sum     summarize_all.py     collects results/headline.json and results/hardware.md
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
STAGES = {
    'tune': [['tune_shots.py', s] for s in 'abcdef'],
    'shots': [['shots_val.py']],
    'hw': [['hardware.py', '--workers', '2']],
    'swapnet': [['swapnet.py', '--n', '11']],
    'e2e': [['e2e.py', '--n', '20', '--mode', 'refit', '--tau', '0.1', '--p', '2', '--e2', '0.003', '--shots', '1024',
             '--workers', '2']],
    'sum': [['summarize_all.py']],
}


def main():
    todo = sys.argv[1:] or ['tune', 'shots', 'hw', 'swapnet', 'e2e', 'sum']
    for st in todo:
        for cmd in STAGES[st]:
            print('>>', ' '.join(cmd), flush=True)
            subprocess.run([sys.executable, str(HERE / cmd[0])] + cmd[1:], check=True, env=ENV, cwd=HERE.parents[1])


if __name__ == '__main__':
    main()
