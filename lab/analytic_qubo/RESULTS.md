# Analytic (closed-form / local) QUBO: results

**Track.** This track asks whether a first-principles QUBO, built without the 256-schedule cost table, can stand in for the
current adaptive projection, and how it compares as an ablation. The derivation is in `DERIVATION.md`, written to paste into the report.

**Status.** Every number below was recomputed in a second session, and the reruns match the first run exactly. The logs are in
`results/rerun/`.

## Reproduce

```
python lab/analytic_qubo/reproduce.py --qaoa --seeds   # closed-form check, recursion count, surrogate table, QAOA current vs analytic on VAL, R seeds 0-4 (~8 min, 2 workers)
python lab/analytic_qubo/qaoa_eval.py --set val --rules uniform_repo red local_gibbs   # QAOA ablation rows (~6.5 min)
python lab/analytic_qubo/local_projection_count.py     # simulation count of the CURRENT projection, rebuilt from local tables (~10 s)
python lab/analytic_qubo/coef_compare.py               # analytic vs adaptive Ising coefficients; prints R's analytic QUBO (~10 s)
python lab/analytic_qubo/clip_sensitivity.py           # post-hoc sensitivity to the two shape constants (~1 min)
```

**Builder.** `analytic_qubo.build_analytic_qubo(s, rule='webster') -> (offset, Q)`, with E = offset + xᵀQx and the same bit order as the repo.

**Protocol.**
- QAOA is the repo's `run.solve`, unchanged: p = 4, split γ, per-qubit β, Gibbs η = 2, 40 + 260 COBYLA evaluations including
  probes (max 300 asserted), 4,096 shots, sampling seed 0.
- Only `qubo.build_qubo` is swapped, inside the worker process.
- The metric is the 4,096-shot sampled metric that `run.solve` computes.
- The current config is `run.solve(s, 0)` on exactly the same VAL.
- Tuning used only `random_scenarios(60 or 200, seed=31)`.

## Headline: VAL = R + `random_scenarios(60, seed=555)` (61 scenarios)

| QUBO | regret ≤ 5 % | regret = 0 | max regret | rubric pts /10 | mean Pearson | mean lift | min lift | mean enrich. | P(opt) | R lift / enrich. (seed 0) |
|---|---|---|---|---|---|---|---|---|---|---|
| **current: adaptive projection** | 0.984 | 0.721 | 5.2 % | 10.00 | 0.9889 | **0.9070** | **0.7336** | **16.79** | **0.343** | 0.8823 / 15.56 |
| **analytic, Webster point (candidate)** | 0.984 | 0.590 | 5.2 % | 10.00 | 0.9901 | 0.9020 | 0.7183 | 16.56 | 0.317 | **0.8877** / 15.13 |
| analytic, local per-node Gibbs measure | 0.984 | 0.754 | 5.2 % | 10.00 | 0.9868 | 0.9045 | 0.7115 | 16.57 | 0.342 | 0.8879 / 14.87 |
| analytic, p* = ½ (= repo uniform Walsh) | 0.967 | 0.574 | 11.8 % | 9.89 | 0.9917 | 0.8951 | 0.7174 | 16.18 | 0.303 | 0.8470 / 13.26 |
| pairwise physics only (all-red Möbius, order 2) | 0.377 | 0.262 | 73.5 % | 6.30 | 0.9279 | 0.7787 | 0.4648 | 10.69 | 0.151 | 0.7810 / 12.70 |

**Paired difference, Webster − adaptive, on VAL** (mean ± s.e.):

| metric | mean difference | Webster better | Webster worse |
|---|---|---|---|
| lift | −0.0049 ± 0.0023 | 25 | 36 |
| enrichment | −0.23 ± 0.09 | 26 | 35 |
| P(opt) | −0.026 ± 0.014 | — | — |
| exact-distribution lift | −0.0052 ± 0.0023 | — | — |
| regret | — | 0 | 11 |

All 11 scenarios where Webster has higher regret stay at ≤ 4.9 %.

**Scenario R, sampling seeds 0–4** (training is deterministic, so seeds only change shot noise):

| QUBO | lift min | lift mean | enrichment mean | P(opt) mean |
|---|---|---|---|---|
| adaptive | 0.8732 | 0.8785 | 15.36 | 0.414 |
| Webster | **0.8840** | **0.8869** | 15.12 | **0.465** |

On R, both QUBO minimizers are `11101010`, cost 72 = C\*.

**Circuit size.** The A4–B couplings are structurally zero, so there are at most 24 couplings instead of 28 (22.8 nonzero on
average on VAL). On R this gives 96 instead of 112 RZZ gates at p = 4. In the adaptive QUBO these four couplings are tiny
(≤ 7·10⁻⁵ of max |J|) but nonzero.

## Coefficient agreement (`coef_compare.py`, VAL)

The analytic Ising vector (h, J) has Pearson **0.998** (min 0.989) with the adaptive one, and relative L2 distance **8.0 %**.
For the uniform Walsh projection the figures are 0.997 and 9.7 %.

The analytic expansion therefore reproduces the current QUBO almost coefficient by coefficient. Every term gets a traffic reading:
- A–A and B–B: interference plus the exact switching term;
- A1..A3 × B: spillback and green-wave;
- A4 × B: exactly zero.

## Simulation counts: what the CURRENT projection needs

`local_projection_count.py` rebuilds both repo projections from local tables only and matches `qubo.build_qubo` with
**max energy error 0.0** on all 61 VAL scenarios.

| construction | single-approach 4-slot recursions | sums over the joint 2^8 space? |
|---|---|---|
| repo code as written (`approach_costs` on 256 joint schedules) | 1,024 | yes |
| current projection, local windows as in the brief (16 + 16 + 16 + 256) | **304** | — |
| same, using that the linked B approach does not see A4 (16 + 16 + 16 + 8×16) | **176** | — |
| uniform projection | 176 | no: it factorises per approach |
| **adaptive** projection | 176 | **yes**: non-product Gibbs weights; its normal equations sum the assembled cost Σ_k D_k over all 256 joint schedules |
| **analytic Webster rule** | **176** | **no**: product measure, per-approach sums of ≤ 2^7 terms; C(x) is never assembled |
| pure closed-form pairwise model | 62 explicit formula evaluations | no |

*Correction to the first draft:* it said the adaptive projection "needs the joint table" in the sense of joint simulations. That
was imprecise. It needs no more simulator calls than the analytic rule. What it does need is a sum of the assembled 256-entry
cost over the joint space, under weights from the surrogate's own energies. At 8 bits that sum is trivial, but it is exactly
what the analytic rule avoids. That is what makes the analytic rule scale linearly in the number of approaches on longer corridors.

## Robustness of the two shape constants (`clip_sensitivity.py`, surrogate level, post hoc, no re-selection)

| rule | tune (n=200): ≤ 5 % / max regret | VAL: ≤ 5 % / max regret |
|---|---|---|
| clip 0.15 | 0.975 / 43.6 % | 0.984 / 5.2 % |
| clip 0.20 | 0.980 / 7.7 % | 0.984 / 5.2 % |
| **clip 0.25 (frozen)** | **0.985 / 5.9 %** | 0.984 / 5.2 % |
| clip 0.30 / 0.35 | 0.980 / 11.0 % | 0.984 / 5.2 % |
| clip 0.40 | 0.965 / 11.0 % | 0.984 / 5.2 % |
| clip 0.25, no platoon demand | 0.970 / 9.1 % | 0.984 / 5.2 % |
| clip 0.50 (= uniform) | 0.920 / 19.5 % | 0.967 / 11.8 % |
| repo adaptive | 0.980 / 11.0 % | 0.984 / 5.2 % |

Any interior clip in [0.2, 0.4] matches the adaptive projection on regret. The expansion point must stay interior:
clip 0.15 produces a 43.6 % outlier.

## Tuning history (seed 31 only)

**Full QAOA, 60 scenarios, seed 0:**

| QUBO | mean lift | min lift | mean enrich. | regret ≤ 5 % | pts /10 |
|---|---|---|---|---|---|
| adaptive (current) | 0.8932 | 0.7314 | 16.61 | 0.967 | 9.94 |
| Webster v0 (clip 0.3, no platoon demand) | 0.8908 | 0.7242 | 16.66 | 0.933 | 9.90 |
| **Webster final (clip 0.25, platoon-aware demand)** | 0.8915 | 0.7069 | 16.66 | 0.983 | 9.99 |
| local per-node Gibbs | 0.8951 | 0.6939 | 16.64 | 0.950 | 9.92 |
| uniform | 0.8786 | 0.7075 | 16.21 | 0.867 | 9.58 |
| all-red pairwise | 0.7575 | −0.084 | 10.77 | 0.367 | 6.15 |

**Variants tried and dropped:**
- *Fluid closed-form differences* (fractional greens): Pearson 0.96, regret 7–11 %.
- *Möbius anchored at the greedy plan or at all-0 / all-1*: 12–18 % mean regret.
- *Mean-field self-consistent expansion point* and *Webster × local Gibbs hybrid*: no gain on sharp Gibbs proxies.

## Verdict

The cost-table-free analytic QUBO **ties** the adaptive projection on surrogate regret: 98.4 % ≤ 5 %, full 10/10 rubric points,
max 5.2 % on VAL. Its Pearson is higher (0.990 vs 0.989). Its QUBO minimizer is exactly optimal less often (59 % vs 72 %), and that
costs a small amount of QAOA sampling quality: −0.005 lift (about 2 s.e.), −0.23 enrichment and −0.026 P(opt) on VAL. On the
tuning set the lift gap is −0.002, which is not significant. It is better on R robustness (min lift 0.884 vs 0.873) and uses 16
fewer RZZ gates.

**Recommendation:** keep the adaptive projection in the main pipeline, and use this work in two ways:
1. as the **derivation backbone** in the report. Steps 1–3 of `DERIVATION.md` give every coefficient a traffic meaning, and the
   adaptive QUBO is reproduced at coefficient Pearson 0.998;
2. as an **ablation row**: a QUBO that never assembles the joint cost reaches the same regret and about 99.5 % of the lift.

Swapping it into `run.py` is defensible if the team values a fully factorised derivation over about 0.5 % mean lift. Integration
would mean copying `analytic_qubo.py` and calling `build_analytic_qubo(s)` in place of `qb.build_qubo` in `run.solve`.

## Files
- `analytic_qubo.py`: `build_analytic_qubo`, Taylor coefficients, Webster point, ablation rules
- `closed_form.py`: closed-form pairwise physics plus a brute-force check (max error 0)
- `qaoa_eval.py`: full-pipeline evaluator with the swapped QUBO
- `reproduce.py`: headline numbers
- `local_projection_count.py`, `coef_compare.py`, `clip_sensitivity.py`: second-session additions
- `results/*.json`: every row
- `results/rerun/*.txt`: second-session logs
- `explore*.py`, `diag.py`, `check.py`: tuning-set exploration only (seed 31)
