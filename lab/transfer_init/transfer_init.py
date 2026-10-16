"""Run-time side of the parameter-transfer initialisation (lab/transfer_init).

The transfer vectors are scenario-independent constants stored in transfer_angles.json; they are regenerated
deterministically by `python lab/transfer_init/learn_angles.py` (learning set = random_scenarios(150, seed=31)
only). This module only *reads* them and defines the candidate / ablation configurations of common.train_ti.

    CANDIDATE_CFG       the proposed pipeline change (init='transfer+ramp')
    ABLATION_VARIANTS   clean initialisation ablation: identical trainer, only the start differs
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANGLES = HERE / 'transfer_angles.json'


def angles(entry='split_qubit_eta2'):
    """Learned transfer vector (qaoa.Ansatz parameter layout) of one entry of transfer_angles.json."""
    return list(json.loads(ANGLES.read_text())[entry]['theta'])


def resolve(cfg):
    """Replace 'json:<entry>' strings by the stored vectors."""
    cfg = dict(cfg)
    for k in ('transfer', 'transfer_base'):
        v = cfg.get(k)
        if isinstance(v, str) and v.startswith('json:'):
            cfg[k] = angles(v.split(':', 1)[1])
    return cfg


# p=4, cost_mode='split', mixer_mode='qubit', Gibbs eta=2 (= repo final config); only the initialisation changes.
CANDIDATE_NAME = 'transfer+ramp (split/qubit p=4, eta=2)'
CANDIDATE_CFG = dict(init='transfer+ramp', transfer='json:split_qubit_eta2')

ABLATION_VARIANTS = [
    # current repo initialisation (3 linear-ramp probes, standard-ansatz COBYLA to eval 40, rich COBYLA to 300)
    ('init ablation: ramp (= run.solve, re-implemented)', dict(init='ramp', cache=False)),
    # 3 uniform[0, pi/2) probes of the standard ansatz, seed 0, then the same two stages
    ('init ablation: random (3 probes, seed 0)', dict(init='random', init_seed=0, cache=False)),
    # 1 evaluation at the transfer vector of the rich ansatz, rich COBYLA to 300 (stage 1 skipped)
    ('init ablation: transfer only', dict(init='transfer', transfer='json:split_qubit_eta2')),
]
