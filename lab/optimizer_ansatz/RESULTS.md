# lab/optimizer_ansatz: in-budget optimiser and ansatz study

**Frozen candidate:** multi-angle QAOA, p = 4. Each layer has one angle per field term, one shared coupling
angle and one mixer angle per qubit (68 parameters). It is trained by `qaoa_nft.train`: 3 ramp probes and COBYLA
on standard QAOA up to 10 evaluations, then exact sequential minimal optimisation (NFT/Rotosolve) sweeps
with 16-evaluation COBYLA blocks on the coupling angles, up to 300 evaluations. The choice was made on tuning
seeds 31 and 7 only (40 scenarios each) and frozen before VAL was run.

## Headline: VAL = [scenario R] + random_scenarios(60, seed=555), 61 scenarios

All arms go through the unmodified `run.solve`; only the trainer and the pipeline config are swapped,
in-process. Metric: **4096-shot sampled** lift, enrichment and P(opt), with sampling seed 0
(`metrics.baselines.6_qaoa_raw_samples`), the same for every arm.

| arm | R lift | R enr | R P(opt) | mean lift | min lift | mean enr | min enr | mean P(opt) | regret<=5% | sched. = C* | 1-qubit entropy (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current (split, COBYLA) | 0.882 | 15.56 | 0.424 | 0.907 | 0.734 | 16.79 | 10.82 | 0.343 | 0.984 | 0.721 | 0.229 |
| **candidate field-NFT p=4** | **0.992** | **19.50** | **0.976** | **0.983** | **0.929** | **19.47** | **19.10** | **0.661** | 0.984 | 0.705 | 0.043 |
| field-NFT p=3 | 0.998 | 19.65 | 0.997 | 0.986 | 0.922 | 19.54 | 19.23 | 0.655 | 0.984 | 0.705 | 0.040 |
| field-NFT p=2 | 0.998 | 19.65 | 0.990 | 0.985 | 0.913 | 19.54 | 18.14 | 0.656 | 0.984 | 0.721 | 0.035 |
| field-NFT p=1 | 1.000 | 19.69 | 1.000 | 0.986 | 0.922 | 19.69 | 19.69 | 0.592 | 0.984 | 0.623 | 0.000 |
| field-NFT p=4, **no couplings** (J:=0 in circuit) | 1.000 | 19.69 | 1.000 | 0.977 | **0.727** | 18.72 | **0.00** | 0.543 | 0.984 | 0.574 | 0 |
| field-NFT p=1, **no couplings** | 1.000 | 19.69 | 1.000 | 0.982 | 0.847 | 19.05 | 0.00 | 0.543 | 0.984 | 0.574 | 0 |
| field ansatz + current COBYLA trainer | 0.861 | 14.99 | 0.341 | 0.874 | 0.722 | 15.47 | 10.32 | 0.239 | 0.984 | 0.721 | 0.298 |
| split ansatz + layerwise INTERP + COBYQA | 0.913 | 16.72 | 0.666 | 0.930 | 0.768 | 17.72 | 12.98 | 0.418 | 0.984 | 0.721 | 0.228 |
| split ansatz + COBYQA stage 2 | 0.891 | 15.89 | 0.504 | 0.927 | 0.774 | 17.53 | 12.95 | 0.395 | 0.984 | 0.721 | 0.220 |

The regret column is a property of the QUBO, so it is the same in every row. "sched. = C*" is the fraction
of scenarios where the decoded raw schedule (the lowest-QUBO-energy sample) is officially optimal.

**Scenario R, sampling seeds 0-4** (training is deterministic and seed-independent; only the shots change):

| arm | lifts | min lift (clipped) | enrichments |
|---|---|---|---|
| current | 0.8823 0.8815 0.8781 0.8732 0.8776 | 0.873 | 15.56 15.43 15.35 15.14 15.35 |
| candidate | 0.9916 0.9921 0.9926 0.9922 0.9926 | **0.992** | 19.50 19.50 19.50 19.51 19.50 |
| split + INTERP + COBYQA | 0.9134 0.9143 0.9160 0.9085 0.9142 | 0.909 | 16.72 16.78 16.70 16.51 16.72 |

## Checks (`checks.py` -> results/checks.json)

- **Budget.** I wrapped `qaoa.Ansatz.statevector` to count every simulation actually run during training.
  On all 61 VAL scenarios the count equals `len(history)`, and it is 299-300 every time. The 3 probes and
  the re-evaluation of the embedded point are included. The NFT value at the new coordinate is computed
  analytically, and that number is never used as a score. The final state is re-simulated after training.
  The tuning harness reports |f_bookkept - f_resimulated| <= 2e-15.
- **The circuit is the trained one.** `circuit.build_circuit(ansatz, params)` with the Qiskit Statevector
  matches the trainer's probabilities to a max |dp| of 4.0e-15 over 61 scenarios. The QASM3 export, loaded
  back, also matches to 2.2e-15. The 4096-shot TV distance is <= 0.006 on the 6 scenarios checked with seed 7.
- **Determinism.** Two `run.solve` calls (R, seeds 0-4) give identical params, history and counts.
- **Parity.** The drop-in `qaoa_nft.train` gives bit-identical params and history to the tuning harness
  (`trainers.train`) for the frozen config.
- **Runtime per seed (one process, scenario R).** Candidate 0.89 s in total (0.71 s training); current
  2.59 s (2.42 s training). On VAL with 2 workers: mean 1.37 s and max 1.92 s, against 3.94 s and 4.96 s.

## Method (precise)

H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j is the Ising form of the QUBO, and sigma = std of H_C over the
256 basis states. The circuit is

|psi(theta)> = prod_{k=1..p} [ prod_i exp(+i b_{k,i} X_i) ] exp(-i [ sum_i g_{k,i} h_i Z_i + g_{k,J} sum_{i<j} J_ij Z_i Z_j ] / sigma) |+>^8

The training objective uses QUBO energies only:
f(theta) = -log sum_x p_theta(x) exp(-eta (E(x) - E_min)/std E), with eta = 2.

**NFT coordinate update.** Take a parameter t that enters as exp(-i t (w/2) P), with P a Pauli string.
Here w = 2 for every mixer angle and w = 2|h_i|/sigma for every field angle. For any diagonal B,

y(t) = <B>(t) = c + a cos(w(t - t0)) + b sin(w(t - t0))

Two evaluations, y+ and y- at t0 +/- pi/(2w), give c = (y+ + y-)/2, b = (y+ - y-)/2 and a = y(t0) - c.
The update is t* = t0 + atan2(b, a)/w, which is the exact 1-D global maximum of <B>, with value
c + sqrt(a^2 + b^2). Since f = -log y is monotone, this is also the 1-D global minimum of f.

The coupling angles g_{k,J} are not single-frequency, so they get short COBYLA blocks. Per seed the budget
is 3 probes + 7 COBYLA evaluations (standard QAOA) + 1 re-evaluation at the embedding, then about two full
sweeps of 64 coordinates x 2 evaluations, with two 16-evaluation coupling blocks. Total: 299-300.

## Tuning-set results (selection only; exact-distribution metrics, 40 scenarios per seed)

| config | s31 lift (min) | s31 enr | s31 P(opt) | s7 lift (min) | s7 enr | s7 P(opt) |
|---|---|---|---|---|---|---|
| current (split, COBYLA 40+260) | 0.891 (0.730) | 16.61 | 0.258 | 0.904 (0.771) | 16.47 | 0.311 |
| split, stage-2 COBYLA rhobeg 0.15 | 0.895 (0.701) | 16.75 | 0.271 | n/a | | |
| split, stage-2 COBYQA r0=0.075 | 0.913 (0.805) | 17.62 | 0.334 | 0.933 (0.863) | 17.75 | 0.369 |
| split, stage-2 Powell | 0.805 (0.690) | 12.91 | 0.133 | n/a | | |
| split, stage-2 Nelder-Mead | 0.770 (0.621) | 10.86 | 0.082 | n/a | | |
| split, stage-1 3 restarts | 0.878 (0.714) | 16.05 | 0.230 | n/a | | |
| split, staged (single cost + qubit mixer, 80 evals) | 0.890 (0.712) | 16.55 | 0.261 | n/a | | |
| split, eta 1 -> 2 schedule | 0.889 (0.714) | 16.65 | 0.250 | n/a | | |
| split, layerwise INTERP 10/20/30/40 + COBYLA | 0.909 (0.773) | 17.34 | 0.299 | 0.920 (0.811) | 17.07 | 0.340 |
| split, layerwise INTERP + COBYQA | 0.931 (0.828) | 18.07 | 0.390 | 0.945 (0.866) | 18.07 | 0.430 |
| split, NFT on mixers + COBYLA16 blocks | 0.903 (0.772) | 17.19 | 0.283 | 0.918 (0.787) | 17.02 | 0.347 |
| field, current COBYLA trainer | 0.863 (0.679) | 15.40 | 0.192 | n/a | | |
| field, COBYQA stage 2 | 0.904 (0.749) | 17.33 | 0.292 | 0.923 (0.818) | 17.42 | 0.346 |
| field-NFT p=4, s1=40 gen8 | 0.980 (0.886) | 19.37 | 0.695 | 0.978 (0.906) | 19.34 | 0.611 |
| field-NFT p=4, s1=40 gen16 | 0.982 (0.887) | 19.42 | 0.700 | 0.981 (0.911) | 19.41 | 0.613 |
| field-NFT p=4, s1=10 gen8 | 0.983 (0.881) | 19.45 | 0.705 | 0.985 (0.950) | 19.41 | 0.668 |
| **field-NFT p=4, s1=10 gen16 (FROZEN)** | **0.986 (0.880)** | 19.52 | 0.712 | **0.987 (0.953)** | 19.47 | 0.674 |
| field-NFT p=4, energy objective | 0.982 (0.881) | 19.41 | 0.688 | n/a | | |
| field-NFT p=4, eta 1 -> 2 | 0.978 (0.882) | 19.37 | 0.649 | n/a | | |
| field-NFT p=4, stage-1 layerwise INTERP | 0.975 (0.878) | 19.28 | 0.617 | 0.975 (0.909) | 19.30 | 0.587 |
| field-NFT grow 1->4 at 150/200/250 (old; layers 3-4 stay identity) | 0.977 (0.798) | 19.43 | 0.638 | 0.981 (0.838) | 19.50 | 0.533 |
| field-NFT grow2, new layer first, 60/130/210 | 0.917 (0.699) | 18.18 | 0.550 | 0.941 (0.752) | 18.53 | 0.487 |
| field-NFT grow2, new layer first, 100/170/240 | 0.954 (0.738) | 18.99 | 0.605 | 0.972 (0.807) | 19.28 | 0.525 |
| field-NFT grow2, new layer first, 150/200/250 | 0.977 (0.798) | 19.43 | 0.638 | 0.981 (0.838) | 19.51 | 0.533 |
| field-NFT p=1 / p=2 / p=3 | 0.990 / 0.983 / 0.985 | | | 0.989 / 0.987 / 0.985 | | |
| field-NFT p=4, no couplings | 0.976 (0.783) | 19.69 | 0.575 | 0.985 (0.839) | 19.69 | 0.550 |

**Selection rule:** among configurations that really train all p = 4 layers, pick the best mean lift
averaged over both tuning seeds. On both seeds, s1=10 beat s1=40 and gen16 beat gen8, and the combination
was best on both seeds. It was then frozen. The full log is in `results/tune.jsonl`.

## What the per-field angles and the entangling layers do (an honest answer to "is this still QAOA?")

1. **Every computational basis state can be reached at p = 1.** Set g_{1,i} h_i/sigma = -/+ pi/4 and
   b_{1,i} = pi/4. Then exp(i pi/4 X) exp(-/+ i pi/4 Z)|+> = |0> or |1> (up to phase) on each qubit. So
   the per-field ansatz contains the QUBO ground state exactly. This is where most of the gain comes from:
   1-qubit expressivity combined with an optimiser that can use it. The same ansatz trained by the current
   COBYLA trainer is *worse* than current (0.874 vs 0.907), because 68 coupled parameters cannot be trained
   in 300 COBYLA evaluations. NFT's exact coordinate updates are what make it trainable.
2. **The trained states are close to product states.** Mean single-qubit entanglement entropy is 0.043 bits,
   against 0.229 for the current pipeline. At p = 1 the optimiser switches the coupling off (entropy ~5e-6).
3. **Removing the ZZ layers (J := 0 in the circuit, same trainer, same QUBO objective).** At p = 4 the mean
   lift barely moves (0.977 vs 0.983). The uncoupled ansatz wins small margins on 47/61 scenarios, because
   it can put all its mass on one basis state. It fails badly on a few scenarios, where the product-state
   (mean-field) landscape traps it in a wrong basis state. The worst cases are rand0 (+0.23 lift with
   couplings), rand24 (+0.15) and rand36 (+0.12). Overall, min lift 0.727 -> 0.929 and min enrichment
   0.00 -> 19.10. So **the entangling layers add worst-case robustness, not mean performance.** The
   enrichment-0 failures of the product ansatz would cost the worst-seed component directly.
4. **Is it still QAOA?** It is multi-angle QAOA (Herrman et al. 2022) restricted to one shared ZZ angle per
   layer. It is still built from the QUBO's own cost unitaries and an X mixer. Stage 1 is plain standard
   QAOA, and the stage-2 starting point is exactly that standard-QAOA state. Results 1-3 must be disclosed
   in the report: on this 8-variable problem the advantage over the current ansatz is not an entanglement
   effect.

## Negative and neutral results

- **Layer-wise growth p=1->4** (identity-initialised new layers, NFT): never better than training p = 4
  directly. The earlier "best" growth run (150/200/250) has a defect: its sweep restarts at layer 1, so in
  the 50-evaluation phases layers 3-4 are never touched. That leaves 2.0 identity layers on average (see the
  `identity layers` lines in `results/robust_*.log`), so it was really a p = 1-2 circuit labelled p = 4.
  Sweeping the new layer first fixes the bookkeeping but does not improve lift. INTERP-initialised layerwise
  standard QAOA in stage 1 helps the split ansatz (+0.018 on s31, +0.017 on s7) but not the field-NFT one
  (-0.005 on s31, -0.003 on s7).
- **Depth:** under the fixed 300-evaluation budget, depth is roughly flat (p=1..4 lift 0.986 / 0.985 /
  0.986 / 0.983 on VAL). p=4 has the best worst-case lift (0.929).
- **Optimisers on the current split ansatz:** Powell and Nelder-Mead are much worse. COBYLA restarts,
  staged mixer-first training and eta schedules (1 -> 2) are neutral. COBYQA is +0.02. Layerwise INTERP
  combined with COBYQA is +0.023 on VAL (0.930). That is the best gain available *without changing the
  ansatz*.
- **Objectives:** the energy objective and eta in {1, 4} are within +/-0.005 of Gibbs eta=2 under field-NFT.

## Integration (drop-in)

1. Copy `lab/optimizer_ansatz/qaoa_nft.py` to the repo root. It imports only `numpy`, `scipy` and `qaoa`.
2. In `run.py`, add `import qaoa_nft`. In `solve()`, replace `qaoa.train(` with `qaoa_nft.train(`. The
   keyword arguments are the same: p, cost_mode, mixer_mode, kind, eta, max_evals, ramp_dts, stage1_evals,
   init, init_seed.
3. In `run.DEFAULTS`, set `cost_mode='field'` and `stage1_evals=10`. Everything else stays: p=4,
   mixer_mode='qubit', kind='gibbs', eta=2.0, measure='adaptive', kappa=1.0, ramp_dts=(0.75, 1.0, 1.25),
   init='ramp', max_evals=300. The trainer defaults gen_evals=16 and gen_rho=0.05 are the frozen values.
4. `experiments/noise.py::trained_circuit`: use `qaoa_nft.train` with `stage1_evals=cfg['stage1_evals']`
   so the noise study uses the new final circuit.
5. The ablations stay runnable. `kind='cvar'` and the fully standard ansatz (`cost_mode='single',
   mixer_mode='single'`) fall back automatically to `qaoa.train`. Rows worth adding:
   - "previous final (split + COBYLA)": requires a `trainer` switch in `solve()`.
   - "no entangling layer": pass `np.zeros_like(J)` to the trainer when `cfg.get('no_coupling')`; the
     circuit then contains no RZZ.
   - "field ansatz + COBYLA".

   The numbers for all of these are in the VAL table above.
6. `tests/test_pipeline.py` needs no change. The candidate passes the same assertions: determinism, 4096
   shots, <= 300 evaluations, QASM round trip and TV < 0.06.

## Reproduce

```
python lab/optimizer_ansatz/final_eval.py --quick         # current vs candidate on VAL + R seeds 0-4 (~3 min, 2 workers)
python lab/optimizer_ansatz/final_eval.py                 # all 10 arms above (~12 min)
python lab/optimizer_ansatz/analyze_coupling.py           # coupled vs uncoupled per scenario (needs final_val.json)
python lab/optimizer_ansatz/checks.py                     # budget / circuit / determinism / parity / runtime
python lab/optimizer_ansatz/experiment_robust.py robust confirm --set tune40     # tuning seed 31
python lab/optimizer_ansatz/experiment_robust.py robust confirm nocoup --set tune40s7  # tuning seed 7
```

Files: `qaoa_nft.py` (drop-in trainer), `trainers.py` / `trainers2.py` (all explored trainers), `experiment*.py`
(tuning harness), `final_eval.py`, `checks.py` and `analyze_coupling.py` (VAL, checks, ablation analysis), and
`results/` (`final_val.json`, `checks.json`, `coupling_analysis.json`, `tune.jsonl`, logs). `val_eval.py` is an
earlier, never-run draft that `final_eval.py` supersedes.
