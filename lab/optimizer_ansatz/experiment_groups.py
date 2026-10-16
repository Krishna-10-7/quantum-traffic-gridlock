"""Config groups explored on the tuning set (registered into experiment.GROUPS)."""
from experiment import group  # noqa: F401


@group('nft1')
def g_nft1():
    out = []
    for ge in (8, 12, 24, 32):
        out.append((f'nft split gen{ge}', {'stage2': 'nft', 'gen_evals': ge}))
    for gr in (0.025, 0.1):
        out.append((f'nft split gen16 rho{gr}', {'stage2': 'nft', 'gen_evals': 16, 'gen_rho': gr}))
    out.append(('nft split gen16 gen_first', {'stage2': 'nft', 'gen_evals': 16, 'gen_first': True}))
    out.append(('nft split gen16 backward', {'stage2': 'nft', 'gen_evals': 16, 'nft_order': 'backward'}))
    out.append(('nft field gen8', {'stage2': 'nft', 'cost_mode': 'field', 'gen_evals': 8}))
    out.append(('nft field gen16', {'stage2': 'nft', 'cost_mode': 'field', 'gen_evals': 16}))
    out.append(('nft field gen0', {'stage2': 'nft', 'cost_mode': 'field', 'gen_evals': 0}))
    return out


FIELD = {'stage2': 'nft', 'cost_mode': 'field', 'gen_evals': 8}


@group('field1')
def g_field1():
    out = []
    for s1 in (10, 25, 60):
        out.append((f'field-nft s1={s1}', {**FIELD, 'stage1_evals': s1}))
    for eta in (1.0, 4.0):
        out.append((f'field-nft eta={eta}', {**FIELD, 'eta': eta}))
    out.append(('field-nft eta1->2 (stage1 eta=1)', {**FIELD, 'eta_stage1': 1.0}))
    out.append(('field-nft energy objective', {**FIELD, 'kind': 'energy'}))
    out.append(('field-nft backward', {**FIELD, 'nft_order': 'backward'}))
    out.append(('field-nft gen_rho0.1', {**FIELD, 'gen_rho': 0.1}))
    out.append(('field-nft gen_rho0.025', {**FIELD, 'gen_rho': 0.025}))
    out.append(('field-nft gen6', {**FIELD, 'gen_evals': 6}))
    out.append(('field-nft gen12', {**FIELD, 'gen_evals': 12}))
    out.append(('field COBYLA (current trainer, cost_mode=field)', {'trainer': 'current', 'cost_mode': 'field'}))
    return out


@group('abl')
def g_abl():
    out = [('field-nft NO couplings (product-state ansatz)', {**FIELD, 'no_coupling': True})]
    for p in (1, 2, 3):
        out.append((f'field-nft p={p}', {**FIELD, 'p': p, 'stage1_evals': 10 + 5 * p}))
    out.append(('split-nft p=2', {'stage2': 'nft', 'gen_evals': 16, 'gen_rho': 0.1, 'p': 2}))
    return out


@group('opt')
def g_opt():
    """Stage-2 optimiser comparison on the CURRENT ansatz (split cost, per-qubit mixer)."""
    out = []
    for r2 in (0.03, 0.15, 0.3):
        out.append((f'split cobyla rho2={r2}', {'rho2': r2}))
    for m, r2 in (('cobyqa', 0.075), ('cobyqa', 0.3), ('powell', 0.1), ('nelder-mead', 0.075)):
        out.append((f'split {m} rho2={r2}', {'stage2': m, 'rho2': r2}))
    out.append(('split cobyla, stage1 layerwise INTERP 10/20/30/40', {'layerwise': (10, 20, 30, 40)}))
    out.append(('split cobyla, stage1 3 restarts', {'restarts': 3, 'stage1_evals': 60}))
    out.append(('split cobyla, stage1_evals=80', {'stage1_evals': 80}))
    out.append(('split cobyla, stage1_evals=20', {'stage1_evals': 20}))
    out.append(('split cobyla, mid (single,qubit) 80 evals', {'mid': ('single', 'qubit'), 'mid_evals': 80}))
    out.append(('split cobyla eta 1 -> 2', {'eta_stage1': 1.0}))
    return out


@group('field2')
def g_field2():
    out = [('field-nft p=4 (gen8, s1=40)', dict(FIELD)),
           ('field-nft p=1 (s1=15)', {**FIELD, 'p': 1, 'stage1_evals': 15}),
           ('field-nft p=2 (s1=20)', {**FIELD, 'p': 2, 'stage1_evals': 20}),
           ('field-nft p=1 eta=4', {**FIELD, 'p': 1, 'stage1_evals': 15, 'eta': 4.0}),
           ('field-nft p=1 eta=1', {**FIELD, 'p': 1, 'stage1_evals': 15, 'eta': 1.0}),
           ('field-nft p=1 energy obj', {**FIELD, 'p': 1, 'stage1_evals': 15, 'kind': 'energy'})]
    for th in ((60, 120, 180), (120, 180, 240), (150, 200, 250)):
        out.append((f'field-nft grow 1->4 at {th}', {**FIELD, 'p': 4, 'stage1_evals': 15, 'grow': th}))
    out.append(('field-nft p=1 NO couplings', {**FIELD, 'p': 1, 'stage1_evals': 15, 'no_coupling': True}))
    out.append(('split-nft p=4 gen16 rho0.1', {'stage2': 'nft', 'gen_evals': 16, 'gen_rho': 0.1}))
    out.append(('field cobyqa p=4 rho0.075', {'cost_mode': 'field', 'stage2': 'cobyqa', 'rho2': 0.075}))
    out.append(('field cobyqa p=1 rho0.075', {'cost_mode': 'field', 'stage2': 'cobyqa', 'rho2': 0.075, 'p': 1,
                                              'stage1_evals': 15}))
    out.append(('current', {'trainer': 'current'}))
    return out
