# Parameter-transfer initialisation (lab/transfer_init)

The repo was not modified. All code and results live in this directory. Repo modules are imported read-only, and so is `lab/optimizer_ansatz/trainers.py`.

## Method

**Offline (training scenarios only).** The learning set is L = `traffic_model.random_scenarios(150, seed=31)`. Fix the run-time ansatz A = (p=4, cost_mode='split', mixer_mode='qubit') and the Gibbs temperature eta = 2. For scenario s, H_C is the Ising form of the QUBO divided by its standard deviation. The training objective is exactly the run-time one:

    f_s(theta) = -log sum_x |<x|psi_s(theta)>|^2 exp(-eta (E_s(x) - min E_s) / std E_s)

The transfer vector is the minimiser of the mean objective:

    theta_bar = argmin_theta (1/|L|) sum_{s in L} f_s(theta)

It is computed with L-BFGS on an exact adjoint gradient of a batched numpy statevector (`common.BatchQAOA`, the same unitary and parameter layout as `qaoa.Ansatz`). The solver is multi-started from linear ramps with dt in {0.5, 0.75, 1.0, 1.25, 1.5}, and the lowest mean objective wins.

- The vectors are stored as constants in `transfer_angles.json`. The 40 numbers for the candidate are in entry `split_qubit_eta2`.
- `python lab/transfer_init/learn_angles.py` regenerates them deterministically. `--verify split_qubit_eta2` recomputes without writing and gives a max |diff| of 0.0 against the stored vector.
- Only QUBO energies are used. No official cost, no VAL scenarios and no seeds 2026/99/555 are involved.

**Run time** (`common.train_ti`, called through `transfer_init.py`):

- **`init='transfer'`:**
  1. Evaluate theta_bar once on the rich 40-parameter ansatz.
  2. Run COBYLA (rhobeg 0.075, as stage 2 of `qaoa.train`) on the rich ansatz for the remaining 299 evaluations.
  3. Skip stage 1 (the standard-QAOA stage).
- **`init='transfer+ramp'` (the candidate fixed before VAL):** also evaluates the 3 ramp probes as a fallback. If a ramp probe beats theta_bar, the run continues exactly as the current pipeline. The fallback fired in 0/60 DEV and 0/61 VAL scenarios (`fallback_count.py`), so in practice this is transfer with 3 wasted evaluations.

Every objective evaluation is counted, including the transfer probe, and the total is asserted to be ≤ 300. A point COBYLA has already evaluated is served from a memo and is not re-evaluated.

**What the learned vector looks like.** It is an annealing-like schedule:

- the field and coupling angles both rise over the four layers: gamma_h = (0.94, 1.29, 2.39, 2.98) and gamma_J = (0.40, 1.27, 1.75, 2.71);
- the per-layer mean mixer angle falls: (0.65, 0.44, 0.26, 0.20);
- vectors learned at eta = 1, 3 and 4 differ by less than 0.25 rad in every coordinate.

**Concentration finding** (`logs_learn.txt`, `cache/optima_split_qubit_eta2.npy`). Per-instance optima, taken by L-BFGS from the current pipeline's angles, reach a mean f of 0.004 but do **not** concentrate:

- their median distance to theta_bar is 7.4 rad;
- per-coordinate IQRs are up to 3.3 rad;
- the coordinate-wise median of the optima is useless (mean f = 1.99);
- by comparison, theta_bar reaches mean f = 0.379, and the current pipeline after 300 evaluations reaches 0.212.

So averaging per-instance optima fails, and minimising the mean objective directly is what works.

## Selection (DEV = random_scenarios(60, seed=7), exact-distribution metrics)

| run-time strategy (300 evals unless stated) | mean lift | min lift | mean enrich. | P(opt) |
|---|---|---|---|---|
| current (ramp, run.solve) | 0.9076 | 0.771 | 16.63 | 0.331 |
| random init (3 uniform probes, seed 0) | 0.8583 | 0.633 | 14.87 | 0.259 |
| **transfer only** | **0.9463** | 0.852 | 18.22 | 0.437 |
| transfer+ramp (candidate) | 0.9460 | 0.852 | 18.21 | 0.437 |
| transfer, rich-stage rhobeg 0.03 / 0.15 | 0.9442 / 0.9414 | 0.838 / 0.830 | 18.16 / 18.12 | |
| transfer vector of standard ansatz replacing ramp probes (stage 1 + 2 kept) | 0.9319 | 0.822 | 17.85 | 0.384 |
| transfer + 20-eval global rescale stage / 40-eval per-layer rescale | 0.9452 / 0.9441 | 0.801 / 0.865 | 18.24 / 18.27 | |
| transfer, Gibbs eta = 1 / 3 / 4 (each with its own learned vector) | 0.9428 / 0.9458 / 0.9462 | 0.844 / 0.836 / 0.843 | 18.12 / 18.29 / 18.31 | 0.40 / 0.46 / 0.47 |
| transfer, cost_mode='field' (68 params), eta 2 / 3, COBYLA | 0.9417 / 0.9416 | 0.779 / 0.783 | 18.09 / 18.10 | |

The rhobeg rows come from `logs_tune_basic.txt` and use a preliminary copy of the vector (`cache/explore_bar_split_qubit.npy`, which differs from the stored vector by at most 3e-4). All other rows use `transfer_angles.json`.

**Budget curve (DEV).** These numbers measure how many evaluations the transfer start saves.

| budget | 1 (zero-shot, untrained) | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|
| transfer start | 0.829 | 0.881 | 0.907 | 0.935 | 0.946 |
| ramp start (current) | | | 0.817 | 0.877 | 0.908 |

Transfer with 100 evaluations matches the current pipeline at 300. The single untrained evaluation, at 0.829, already beats the standard ansatz trained for 300 evaluations, which reaches 0.815 on the repo's ablation set.

**Answers to the track questions:**

- Use the mean-objective vector, not the median of per-instance optima.
- Transfer should *replace* the ramp probes and stage 1. Adding the ramp probes costs 3 evaluations and never wins.
- Give the whole remaining budget to the rich-ansatz COBYLA with rhobeg 0.075. Extra rescaling stages do not help.
- Eta is flat in [2, 4], so keep eta = 2.
- With COBYLA, the field ansatz does *not* train better from a good start (0.942 vs 0.946). It only pays off with the NFT trainer (see below).

## Headline: VAL = [scenario R] + random_scenarios(60, seed=555)

Metric: official cost of the 4,096 raw shots of the final bound Qiskit circuit, `StatevectorSampler(seed=0)`, as in `run.solve`. The current arm is `run.solve(s, 0)['metrics']` itself, with unmodified repo code.

| variant | R lift | R enrich. | mean lift (60) | min lift (60) | mean enrich. (60) | min enrich. | mean P(opt) | exact mean lift |
|---|---|---|---|---|---|---|---|---|
| current (run.solve) | 0.882 | 15.56 | 0.9074 | 0.734 | 16.81 | 10.82 | 0.342 | 0.9078 |
| **candidate transfer+ramp** | **0.939** | **18.00** | **0.9396** | **0.773** | **18.08** | **12.58** | **0.445** | 0.9403 |
| transfer only | 0.939 | 18.00 | 0.9404 | 0.776 | 18.10 | 12.63 | 0.446 | 0.9407 |

- Paired over the 60 random scenarios, the candidate beats current on 57/60, and transfer-only beats it on 59/60. The mean paired gain is +0.032 lift.
- The 10th-percentile lift rises from 0.844 to 0.891.
- Surrogate regret ≤ 5% holds on 98.4% of VAL scenarios. The QUBO is unchanged, so this is identical for all arms.
- Runtime is at most 3.6 s per scenario per seed, and 300 evaluations were used.

**Scenario R, sampling seeds 0–4.** Training is deterministic, so only the shots change.

| variant | lift per seed | min | mean enrich. (min) |
|---|---|---|---|
| current | 0.882 0.882 0.878 0.873 0.878 | 0.873 | 15.36 (15.14) |
| candidate | 0.939 0.941 0.940 0.938 0.938 | **0.938** | **18.00 (17.96)** |
| random init (init seed = run seed) | 0.867 0.937 0.889 0.910 0.876 | 0.867 | 15.47 (14.38) |

**Out-of-family check** (`scenarios/edge`, seed 0, reporting only):

| scenario | current lift (enrich.) | candidate lift (enrich.) |
|---|---|---|
| edge_max | 0.957 (8.65) | 0.963 (10.53) |
| edge_zero | 0.9945 (19.53) | 0.9991 (19.66) |

## Initialisation ablation (VAL, 4,096 shots, seed 0)

The trainer, budget, ansatz and objective are identical in all three arms; only the start differs.

| start | R lift | mean lift (60) | min lift | mean enrich. | P(opt) |
|---|---|---|---|---|---|
| random: 3 uniform[0, pi/2) probes, stage 1 + stage 2 | 0.867 | 0.8597 | 0.611 | 15.10 | 0.266 |
| ramp: 3 linear-ramp probes, stage 1 + stage 2 (current) | 0.882 | 0.9074 | 0.734 | 16.81 | 0.342 |
| transfer: 1 evaluation at theta_bar, rich COBYLA | 0.939 | 0.9404 | 0.776 | 18.10 | 0.446 |

The ramp re-implementation reproduces `run.solve` exactly (0.9074 / 16.81).

## On top of the NFT/Rotosolve trainer (lab/optimizer_ansatz), reported separately

`nft_transfer.py` uses the optimizer track's `trainers.train`, `train_grow` and `_nft_loop` unchanged. Only the start of the field/qubit stage differs: ramp probes plus standard-QAOA COBYLA, or 1 evaluation at a transfer vector learned the same way for that depth (`field_qubit_eta2`, `field_qubit_p1_eta2`, `field_qubit_p2_eta2`).

| arm | DEV exact mean lift (min) | VAL sampled mean lift (min) | VAL mean enrich. (min) | VAL R lift |
|---|---|---|---|---|
| field-NFT p=4, ref | 0.9788 (0.903) | 0.9776 (0.923) | 19.30 (18.44) | 0.989 |
| field-NFT p=4, **transfer** | **0.9824** (0.906) | **0.9798** (0.926) | **19.41 (19.16)** | 0.987 |
| field-NFT p=1, ref | **0.9894** (0.911) | **0.9879** (0.922) | 19.69 (19.59) | 1.000 |
| field-NFT p=1, transfer | 0.9857 (0.896) | 0.9857 (0.892) | 19.69 (19.69) | 1.000 |
| field-NFT p=2, ref | 0.9857 (0.900) | 0.9839 (0.932) | 19.50 (18.36) | 0.992 |
| field-NFT p=2, transfer | 0.9813 (0.842) | 0.9798 (0.847) | 19.15 (**0.21**) | 0.956 |
| field-NFT grow 1→4 @150/200/250, ref | 0.9837 (0.838) | 0.9824 (0.825) | 19.57 (16.90) | 1.000 |
| field-NFT grow, transfer p=1 start | 0.9838 (0.891) | 0.9825 (0.825) | 19.61 (17.61) | 1.000 |

With the NFT trainer, transfer helps only at p=4 (about +0.003 lift, and a tighter enrichment floor). It is neutral for growth and harmful at p=1 and p=2. At p=2 one VAL scenario falls into a bad basin, with enrichment 0.21. It is not a consistent improvement there. The NFT trainer itself, from either start, dominates the COBYLA candidate (VAL 0.978–0.988 vs 0.940).

## Reproduce

All commands run from the repo root and use 2 worker processes.

| command | produces | time |
|---|---|---|
| `python lab/transfer_init/evaluate_val.py` | headline VAL table, init ablation and R seeds 0–4 → `results/val.md`, `results/val.json` | ~10 min |
| `python lab/transfer_init/nft_transfer.py val` (or `dev`) | NFT ± transfer → `results/nft_*.json` | ~3 min |
| `python lab/transfer_init/tune.py round3 budget` | DEV selection tables | ~25 min |
| `python lab/transfer_init/learn_angles.py` | regenerates all of `transfer_angles.json` | ~25 min; `split_qubit_eta2` alone takes 1 min |
| `python lab/transfer_init/fallback_count.py` | fallback counts and zero-shot lift | ~1 min |
| `python lab/transfer_init/edge_check.py` | out-of-family check | ~1 min |

Logs are in `logs_*.txt`.
