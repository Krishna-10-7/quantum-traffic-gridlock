# Quantum Traffic Gridlock — traffic-derived QUBO and budgeted multi-angle QAOA

*IBM Qiskit Fall Fest 2026 · QC@IITI. All costs below are computed by the official `scorer.py`. Code, outputs and logs are in the repository. Commands to reproduce every number are in `README.md`.*

## 1. Traffic model and variables

There are 8 binary decisions, x = (A₁…A₄, B₁…B₄), with xᵢ = 1 for NS green and 0 for EW green. One bit per intersection-slot
encodes "exactly one phase is green", so **every bitstring is feasible**. No one-hot penalty, slack or auxiliary variable is
needed, and the QUBO has 8 variables mapped to 8 qubits. The official cost decomposes exactly
(verified against the scorer on 1,240 random pairs):

C(x) = Σ_k D_k(x_{S_k}) + w_sw Σ_{n∈{A,B}} Σ_{t=2..4} (x_{n,t} + x_{n,t−1} − 2x_{n,t}x_{n,t−1}),

where D_k = (delay + w_ov·overflow) of approach k ∈ {A-NS, A-EW, B-NS, B-EW}. **Locality** fixes which bits each approach sees:

- the two A approaches see S_k = A bits;
- the unlinked B approach sees S_k = B bits;
- the linked B approach sees all 8 bits, because transfers round(f·departures) carry A's decisions into B one slot later.

Switching is already quadratic and is kept **exactly**.

## 2. QUBO derivation (scenario-independent modelling rule)

**Why a surrogate is necessary.** Each D_k is built from min(queue, capacity) (service) and max(0, queue − storage) (overflow).
As a pseudo-Boolean function it has a unique multilinear expansion D_k = Σ_S ĉ_S Π_{i∈S} zᵢ with zᵢ = 1 − 2xᵢ. Its degree is
up to 4 within an intersection and up to 8 for the linked approach (spillback). Reducing all cubic-and-higher monomials
would need far more than the 8 spare qubits. We therefore approximate each D_k by a quadratic function.

**Rule.** The QUBO is the weighted-L² projection of the exact traffic cost onto quadratic pseudo-Boolean functions:

E = argmin_{g quadratic} Σ_x μ(x) (C(x) − g(x))², with g(x) = offset + Σ_{i≤j} Q_ij x_i x_j.

- **Uniform μ (Walsh truncation).** This keeps exactly the Walsh coefficients with |S| ≤ 2, ĉ_S = 2⁻⁸ Σ_x C(x) χ_S(x)
  (orthogonality is asserted in the tests). The terms have a direct traffic meaning:
  - ĉ_{i} is the average marginal cost of giving NS rather than EW green in slot i;
  - same-intersection ĉ_{ij} > 0 encodes diminishing returns (a green spent on an already-emptied queue is wasted, Fig. a);
  - A–B couplings arise only from the linked approach, i.e. downstream spillback.
  Because projection is linear, the A-only approaches contribute no A–B terms.
- **Adaptive μ (used).** We set μ_κ(x) ∝ exp(−(E(x) − min E)/(κ·σ_E)) and solve E = Proj_{μ(E)} C as a fixed point
  (8 iterations, κ = 1, frozen). The rationale is that QAOA puts its probability on low-energy states, so approximation
  error should be smallest there. The weights depend **only on the surrogate's own energies**. The true optimum is never
  computed or used; it is the scorer that judges the result. The weights stay broad: their effective sample size is
  157 of 256 schedules on average (167 on R).

Computing the projection evaluates the traffic recurrence on its inputs: 16 local configurations for 4-bit approaches,
256 for the linked one. That is a modelling transform of the dynamics, not an optimisation. No schedule, coefficient or
penalty is scenario-specific. The Ising form uses x = (1 − z)/2 and is stored in `qubo.json` with the variable↔qubit map.

**Surrogate quality** (QUBO exact minimizer evaluated by the official scorer):

| | regret (QUBO min vs C\*) | regret ≤ 5 % | QUBO min optimal | Pearson(E, C) |
|---|---|---|---|---|
| Scenario R | 0 % (72.0 = C\*) | — | yes | 0.993 |
| 100 unseen scenarios, uniform μ | 1.77 % mean, max 31 % | 88 % | 52 % | 0.992 |
| 100 unseen scenarios, adaptive μ | **0.44 % mean, max 8.6 %** | **98 %** | **67 %** | 0.989 |

On the rubric's 5 %→25 % scale, this is 9.67/10 (uniform) vs **9.97/10** (adaptive) expected surrogate-regret credit.
Scenarios are drawn from the published hidden ranges (arrivals 0–8, capacity 4–6, storage 5–8, w_ov 2–6, w_sw 0.25–1.5,
NS/EW link, f 0.4–0.8) and filtered to C_rand − C\* > 5.

## 3. QAOA design

**Hamiltonian and normalisation.** H_C = Σ hᵢZᵢ + Σ J_ij Z_iZ_j, divided by its spectral standard deviation σ (computed from
h, J). This makes angles transfer across scenarios.

**Ansatz (p = 4).** |ψ⟩ = Π_k U_M(β_k) U_C(γ_k) |+⟩^⊗8, with:

- U_C = exp(−i[γ_k^h Σ hᵢZᵢ + γ_k^J Σ J_ij Z_iZ_j]): separate angles for fields and couplings;
- U_M = Π_i exp(+i β_{k,i} Xᵢ): a **per-qubit (multi-angle) mixer**, with H_M = −ΣX so that |+⟩ is its ground state.

This gives 40 parameters. The circuit uses 8 qubits, 8 H, 32 RZ, 112 RZZ and 32 RX gates (depth 47). Standard QAOA is
the special case of tied angles.

**Initialisation.** Three linear-ramp (Trotterised annealing) schedules with Δt ∈ {0.75, 1.0, 1.25}. Each costs one evaluation.

**Objective.** Gibbs objective −log⟨exp(−η(E − E_min)/σ_E)⟩ with η = 2 (Li et al. 2020). It rewards probability on
low-energy states rather than only the mean. All objectives use **QUBO energies only, never the official cost.**

**Optimiser and budget.** COBYLA, two stages. Stage 1 optimises the standard 8-parameter ansatz up to evaluation 40. Its
optimum is embedded into the 40-parameter ansatz (same state), which is refined to a hard cap of **300 evaluations**,
probes included and asserted.

**Training and sampling.** Training uses an exact numpy statevector that implements the identical unitary. It agrees
with Qiskit's `Statevector` of the exported circuit to ~1e-15, checked on every run. The final 4,096 shots come from
Qiskit's `StatevectorSampler(seed)` on that circuit. Training is deterministic, so the seed only affects shot noise, which
explains the tight seed spread. Runtime is about 2 s per seed.

**Decoding.** The raw-QAOA schedule is the sampled bitstring with the lowest QUBO energy (no scorer used). Separately
labelled **hybrid**: steepest-descent 1-flip local search on the official cost, started from the 8 most frequent samples.

## 4. Results

**Scenario R** (C\* = 72.0, C_rand = 151.25), 5 seeds (0–4):

| method | official cost |
|---|---|
| (1) best alternating `10101010` | 116.0 |
| (2) greedy larger-queue `01111001` | 92.5 |
| (3) exact enumeration `11101010` | **72.0** |
| (4) uniform random expected | 151.25 |
| (5) QUBO exact minimizer (surrogate ceiling) | **72.0** |
| (6) QAOA raw samples: mean E_Q | 81.3–82.1 |
| (6) QAOA decoded schedule (every seed) | **72.0** |
| (7) HYBRID (labelled) | 72.0 |

Over the 5 seeds, lift is 0.873 / 0.879 / 0.882 (min / mean / max) and enrichment is 15.14 / 15.37 / 15.56. The exact
optimum receives 40–42 % of shots.

**Generalization** (100 unseen scenarios, generator seed 2026, frozen hyper-parameters):

- lift: mean 0.919, median 0.926, min 0.775;
- enrichment: mean 17.2, min 10.9 (random = 1, maximum 19.7);
- the exact optimum appears among the 4,096 samples in **100 %** of scenarios;
- fraction of the random→optimal gap closed: alternating 0.66, greedy 0.88, QAOA mean-of-samples 0.92,
  QAOA decoded schedule 0.994, hybrid 1.00.

**Ablations** (`results/ablations.md`; R + 40 unseen scenarios, generator seed 99, 4,096 shots):

| variant (one change from final) | R lift | mean lift | min lift | mean enrich. | mean P(opt) |
|---|---|---|---|---|---|
| **final** (p=4, split γ, per-qubit β, Gibbs η=2, adaptive κ=1) | 0.882 | **0.909** | 0.815 | **16.85** | 0.370 |
| depth p = 1 / 2 / 3 | 0.727 / 0.907 / 0.885 | 0.717 / 0.886 / 0.907 | 0.543 / 0.778 / 0.817 | 10.7 / 16.1 / 16.7 | 0.14 / 0.38 / 0.37 |
| standard ansatz (1 γ, 1 β per layer) | 0.786 | 0.815 | 0.711 | 13.12 | 0.172 |
| per-qubit β, single γ | 0.910 | 0.884 | 0.744 | 16.24 | 0.327 |
| objective ⟨E⟩ / CVaR₀.₂₅ | 0.857 / 0.777 | 0.884 / 0.793 | 0.786 / 0.588 | 15.4 / 13.5 | 0.27 / 0.23 |
| Gibbs η = 1 / 4 | 0.869 / 0.883 | 0.904 / 0.910 | 0.796 / 0.817 | 16.5 / 17.1 | 0.34 / 0.42 |
| QUBO uniform μ (Walsh) / κ = 0.5 / κ = 2 | 0.847 / 0.883 / 0.876 | 0.899 / 0.913 / 0.905 | 0.789 / 0.819 / 0.790 | 16.1 / 17.1 / 16.7 | 0.36 / 0.37 / 0.37 |
| init random angles / near-identity | 0.867 / 0.828 | 0.857 / 0.889 | 0.654 / 0.765 | 14.9 / 15.9 | 0.28 / 0.33 |
| budget 100 / 200 evaluations | 0.778 / 0.847 | 0.809 / 0.872 | 0.689 / 0.773 | 12.6 / 15.4 | 0.16 / 0.27 |

Penalties: the encoding needs no constraint penalty (Section 1). The ablated weighting hyper-parameters are κ (surrogate
measure) and η (objective temperature). The largest gains come from the multi-angle mixer (+0.07 lift), the annealing
initialisation (+0.05 vs random) and the full 300-evaluation budget. η = 4 and κ = 0.5 look marginally better here but
were **not** re-tuned on this reporting set; the frozen values stand.

## 5. Extension: noise analysis and noise-aware design (`experiments/noise.py`)

**Noise model.** The circuit is transpiled to {rz, sx, x, cx}. A 2-qubit depolarizing channel with rate e₂ follows each
cx, a 1-qubit channel with rate e₂/10 follows each sx/x, and each readout bit flips with probability e₂. Probabilities
are computed exactly with Aer density matrices on R plus 8 unseen scenarios.

Mean lift on R + 8 unseen scenarios (exact noisy distributions):

| variant | cx | e₂ = 0 | 10⁻³ | 3·10⁻³ | 10⁻² | 2·10⁻² | 5·10⁻² |
|---|---|---|---|---|---|---|---|
| p = 1 | 56 | 0.716 | 0.705 | 0.683 | 0.613 | 0.523 | 0.319 |
| p = 2 | 112 | 0.879 | 0.851 | **0.799** | 0.640 | 0.466 | 0.178 |
| p = 4 (submitted) | 224 | **0.899** | 0.847 | 0.752 | 0.495 | 0.270 | 0.042 |
| p = 4, pruned τ = 0.1 | 115 | 0.881 | **0.852** | 0.797 | 0.635 | 0.461 | 0.182 |
| p = 2, pruned τ = 0.25 | 35 | 0.797 | 0.788 | 0.771 | **0.712** | **0.636** | **0.452** |

**Finding.** The optimal depth is noise-dependent. p = 4 is best only when noiseless. At e₂ ≈ 3·10⁻³ (current
heavy-hex devices), the shallower p = 2 circuit wins, and pruning weak couplings (|J| < τ·max|J|) trades a little
noiseless lift for fewer CNOTs. On hardware we would therefore pick p and τ from a calibrated e₂ rather than fixing p = 4.

## 6. Limitations

- The quadratic projection cannot represent the min/max hinges exactly. In 2 % of scenarios the QUBO minimizer is 5–9 %
  above C\*, which bounds the decoded schedule (the hybrid repairs it). The lift ceiling of the surrogate itself is ≈ 0.99.
- Training uses exact expectations. With shot-based training, COBYLA would need a larger tolerance or SPSA. We expect lower
  lift at equal budget; this was not measured.
- The dense A–B coupling graph is the main noise cost (224 cx at p = 4). On a heavy-hex layout, SWAP routing would add more.
- The projection evaluates the recurrence on all 2^|S_k| local inputs. For long corridors the same rule applies per
  approach, but the adaptive weights would need Monte-Carlo estimation.
- Quantum advantage is neither claimed nor expected at 8 qubits. Exact enumeration of 256 schedules is trivial and is used
  only as a benchmark.
