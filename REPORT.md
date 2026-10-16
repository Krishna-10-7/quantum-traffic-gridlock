# Quantum Traffic Gridlock — closed-form traffic QUBO and exactly-trained multi-angle QAOA

*IBM Qiskit Fall Fest 2026 · QC@IITI. All costs below are computed by the official `scorer.py`. Every number comes from scripts in the repo (`README.md` lists the commands). Settings were frozen on tuning scenarios (generator seeds 1, 7, 11, 31) before any reported set was run. The reported sets are VAL = R plus seed 555, held-out = seed 2026, and ablations = seed 99.*

## 1. Traffic model and variables

There are 8 binary decisions, x = (A₁…A₄, B₁…B₄), with xᵢ = 1 for NS green. One bit per intersection-slot natively encodes "exactly one phase is green", so **all 256 bitstrings are feasible**. No one-hot penalty, slack or auxiliary qubit is needed: 8 variables map to 8 qubits.

The official cost decomposes exactly (verified against the scorer on 1,240 random pairs):

C(x) = Σ_k D_k(x_{S_k}) + w_sw Σ_{n,t≥2} (x_{n,t} + x_{n,t−1} − 2x_{n,t}x_{n,t−1})

Here D_k = delay + w_ov·overflow of approach k, and it follows the Lindley recursion q_t = (q_{t−1} + a_t + τ_t − c·g_t)⁺ with g = x for NS and 1 − x for EW. **Locality** fixes S_k:

- the A approaches see A₁..A₄;
- the unlinked B approach sees B₁..B₄;
- the linked B approach sees B₁..B₄ plus A₁..A₃, through platoons τ_{t+1} = ⌊f·d^A_t + ½⌋. A₄'s platoon leaves the model, so every A₄–B coupling is exactly 0.

Switching is quadratic and is kept exactly.

## 2. QUBO derivation (closed-form, scenario-independent rule)

**Pairwise physics.** Let A_t be the cumulative arrivals and δ_s = min(c, A_s). Then:

- a green at slot s saves (5 − s)·δ_s vehicle-slots;
- two greens s < t on one approach interfere by +(5 − t)[min(c, A_t) − min(c, A_t − δ_s)] ≥ 0, which is zero while the second green is still saturated;
- an A green releases a platoon P_s that costs (4 − s)·P_s at a red B, while a later B green serves it (the "green-wave" term, −(5 − t)[min(c, A^B_t + P_s) − min(c, A^B_t)] ≤ 0).

These closed forms are verified against brute force with zero error. Truncating this all-red expansion at order 2 drops the cascade when three or more greens compete for the same vehicles, and it reaches regret ≤ 5 % in only 38 % of scenarios. The higher-order interference therefore has to be folded into the quadratic terms.

**Rule.** Let D̄_k(p) = E_{x∼Bernoulli(p)} D_k(x) be the multilinear extension. The QUBO is its second-order expansion at the fluid operating point p*:

E(x) = Σ_k [D̄_k(p*) + Σ_i V_ki (x_i − p*_i) + Σ_{i<j} W_kij (x_i − p*_i)(x_j − p*_j)] + switching

- V_ki = E_{p*}[D_k | x_i = 1] − E_{p*}[D_k | x_i = 0] is the expected cost of turning slot i to NS.
- W_kij is the pairwise interaction above, averaged with the other slots' greens drawn from p*.
- Centred monomials are orthogonal under the product measure, so this is exactly the L²(Bernoulli(p*)) projection of each D_k onto quadratics: p* = ½ gives the uniform Walsh truncation, and a vertex gives the all-red expansion.
- p* is the **Webster equal-saturation split** of green time: p*_{n,t} = clip((N_{n,t} + ½)/(N_{n,t} + E_{n,t} + 1), 0.25, 0.75), with N and E the cumulative NS and EW demands. The linked approach adds the expected platoon f·a^A_{t−1}.

**Integrity.** The rule needs **176 single-approach four-slot recursions** (16 + 16 + 16 + 8 platoon vectors × 16). It never simulates a joint schedule, never forms the 256-entry cost table, and never computes or uses the optimum. A test replaces the scorer and every joint-cost function with traps while the QUBO is built. The result has 24 couplings, so 96 RZZ gates at p = 4.

**Surrogate quality.** The QUBO's exact minimizer is scored by the official scorer. VAL = R + 60 unseen scenarios:

| QUBO rule | regret ≤ 5 % | max regret | minimizer = C\* | Pearson(E, C) |
|---|---|---|---|---|
| **Webster expansion (submitted)** | **98.4 %** | 5.2 % | 59 % | 0.990 |
| weighted-L² projection of full table (reference) | 98.4 % | 5.2 % | 72 % | 0.989 |
| uniform Walsh (p* = ½) | 96.7 % | 11.8 % | 57 % | 0.992 |
| pairwise physics only (all-red) | 37.7 % | 73.5 % | 26 % | 0.928 |

On 100 more held-out scenarios (seed 2026), the submitted rule has regret ≤ 5 % in 97 % of cases (mean 0.61 %, max 8.6 %), worth ≈ 9.96/10 rubric points. Scenario R: the minimizer is `11101010` = C\* = 72.

## 3. QAOA design

**Hamiltonian.** H_C = Σ hᵢZᵢ + Σ J_ij Z_iZ_j, divided by its std σ so that angles are scale-free.

**Ansatz (p = 4, multi-angle; Herrman et al. 2022).** Each layer applies:

- a cost layer exp(−i[Σᵢ g_{k,i} hᵢZᵢ + g_{k,J} Σ J_ij Z_iZ_j]/σ), with one angle per field term and one shared coupling angle;
- a mixer layer Πᵢ exp(+i b_{k,i}Xᵢ), with one angle per qubit and H_M = −ΣX, so |+⟩^⊗8 is its ground state.

That is 68 parameters. Standard QAOA is the tied-angle special case. The circuit has 8 qubits, 32 RZ, 96 RZZ and 32 RX gates.

**Objective.** The Gibbs objective f = −log⟨exp(−η(E − E_min)/σ_E)⟩ with η = 2, computed on **QUBO energies only**.

**Exact coordinate training (NFT/Rotosolve; Nakanishi et al. 2020).** Every mixer angle (frequency w = 2) and every field angle (w = 2|hᵢ|/σ) enters as exp(−i t (w/2) P) with P a Pauli. So for any diagonal B, ⟨B⟩(t) = c + a cos w(t − t₀) + b sin w(t − t₀).

- Two evaluations at t₀ ± π/2w recover (a, b, c).
- The update t* = t₀ + atan2(b, a)/w is the exact 1-D global optimum. Because f is monotone in ⟨B⟩, it also minimises f.

**Schedule.** Three linear-ramp (annealing) probes, then 10 COBYLA evaluations of standard QAOA, then embedding into the multi-angle ansatz (same state). After that, about two NFT sweeps over 64 coordinates, interleaved with 16-evaluation COBYLA blocks on the shared coupling angles. The hard cap of **300 evaluations** includes probes and is asserted by wrapping the simulator itself.

**Training and sampling.** Training uses an exact numpy statevector, identical to the exported circuit (|Δp| ≤ 4·10⁻¹⁵, checked every run). The final 4,096 shots come from Qiskit's `StatevectorSampler(seed)`. Training is deterministic, so the seed only moves shot noise. Runtime is about 1 s per seed.

**Decoding.** The raw schedule is the sampled bitstring with the lowest QUBO energy. Separately labelled **hybrid**: 1-flip descent on the official cost, started from the 8 most frequent samples.

## 4. Results

**Scenario R** (C\* = 72.0, C_rand = 151.25), sampling seeds 0–4. The baselines are:

- (1) alternating `10101010`: 116.0;
- (2) greedy larger-queue `01111001`: 92.5;
- (3) exact enumeration `11101010`: **72.0**;
- (4) uniform random: 151.25;
- (5) QUBO minimizer (surrogate ceiling): **72.0**.

(6) **Raw QAOA:** E_Q = 72.41–72.52, lift **0.993 / 0.994** (min / mean over 5 seeds), enrichment **19.52** (maximum 19.69), with **98.5–98.7 %** of all shots on the optimum. The decoded schedule is 72.0 on every seed. (7) Hybrid: 72.0.

**Held-out generalization** (100 unseen scenarios, seed 2026, frozen settings):

- lift: mean **0.979**, median 0.988, min 0.819;
- enrichment: mean **19.28** (min 0.00);
- P(opt): mean 0.60;
- decoded schedule optimal in 62 % of scenarios;
- fraction of the random→optimal gap closed: alternating 0.66, greedy 0.88, QAOA sample mean 0.979, decoded 0.990, hybrid 1.00.

**Ablations** (one change each; R + 40 unseen scenarios, seed 99, 4,096 shots):

| variant (one change from final) | R lift | mean lift | min lift | mean enrich. | mean P(opt) |
|---|---|---|---|---|---|
| **final** (Webster QUBO, p=4 multi-angle, Gibbs η=2, NFT) | 0.993 | 0.982 | 0.867 | 19.20 | 0.73 |
| depth p=1 | 1.000 | 0.987 | 0.872 | 19.44 | 0.65 |
| depth p=2 | 0.997 | 0.987 | 0.848 | 19.26 | 0.78 |
| depth p=3 | 0.996 | 0.983 | 0.861 | 19.09 | 0.70 |
| no ZZ couplings in circuit (product-state ansatz) | 1.000 | 0.985 | 0.932 | 19.69 | 0.56 |
| previous submission: split cost, COBYLA trainer | 0.888 | 0.901 | 0.795 | 16.35 | 0.35 |
| standard QAOA ansatz, COBYLA | 0.776 | 0.815 | 0.690 | 13.20 | 0.18 |
| field ansatz, COBYLA trainer | 0.834 | 0.870 | 0.745 | 15.28 | 0.25 |
| objective <E> (standard) | 0.995 | 0.984 | 0.864 | 19.16 | 0.70 |
| objective CVaR alpha=0.25 (COBYLA) | 0.807 | 0.799 | 0.671 | 13.35 | 0.22 |
| Gibbs eta=1 | 0.994 | 0.983 | 0.867 | 19.23 | 0.71 |
| Gibbs eta=4 | 0.993 | 0.982 | 0.868 | 19.21 | 0.72 |
| QUBO: adaptive projection of full cost table | 0.992 | 0.987 | 0.948 | 19.46 | 0.77 |
| QUBO: uniform Walsh projection | 0.995 | 0.979 | 0.839 | 19.22 | 0.71 |
| QUBO: pairwise physics only (all-red expansion) | 0.904 | 0.867 | 0.408 | 14.42 | 0.30 |
| init: random angles (3 probes) | 0.997 | 0.974 | 0.848 | 19.02 | 0.71 |
| init: near-identity | 1.000 | 0.989 | 0.849 | 19.65 | 0.71 |
| budget 100 evaluations | 0.921 | 0.928 | 0.836 | 17.43 | 0.63 |
| budget 200 evaluations | 0.988 | 0.977 | 0.861 | 18.97 | 0.71 |

Reading: (i) the NFT trainer is the main gain: +0.08 lift over the previous COBYLA submission and +0.17 over standard QAOA. (ii) The QUBO rules agree within 0.01 mean lift. The full-table projection is a little more robust (min 0.948), and pairwise-only physics is clearly worse. (iii) Depth p = 1–3 already reaches the same level. Removing the ZZ couplings is not worse on this set; on VAL it collapsed once (min enrichment 0). At 8 qubits, the problem is solved to the surrogate ceiling by a near-product state, so no advantage from entanglement is claimed. (iv) The objective (Gibbs vs ⟨E⟩, η) barely matters with exact coordinate updates. CVaR needs COBYLA and is worse. (v) Budget matters up to about 200 evaluations. There is no constraint penalty to ablate (Section 1), so the weighting hyper-parameters were ablated instead.

## 5. Extensions

**(a) Three-intersection chain A→B→C** (`lab/chain_extension/`, 12 qubits).

- *Definition.* Two links l₁ = A→B and l₂ = B→C, each with a phase and a fraction. The slot semantics are copied line by line from the scorer.
- *Reduction.* With C empty and f₂ = 0, the chain cost equals the official cost on all 4,096 schedules (tested on R + 30 scenarios).
- *Light cone.* Approach (n, p), reached through m same-phase links, depends only on the upstream bits in slots 1..4−m, so |S_k| ≤ T(T+1)/2 = 10 for any corridor length. The local QUBO therefore needs Σ 2^|S_k| simulations: 704 vs 4,096 at L = 3, and 5,904 vs 2³² at L = 8.
- *Results.* The 2-node design transfers unchanged (previous COBYLA trainer). On 20 random chains × 5 seeds: lift 0.871 (min 0.753), enrichment 17.8, regret ≤ 5 % in 90 %. Joint 12-bit coordination beats a sequential (A,B)-then-C optimisation, which misses the optimum in 13/20 scenarios.

**(b) Noise and hardware** (`experiments/noise.py`, `lab/shots_hardware/`). Exact density matrices with 2q depolarizing rate e₂ (1q rate e₂/10, readout flip e₂):

| variant | cx | e₂ = 0 | 10⁻³ | 3·10⁻³ | 10⁻² | 2·10⁻² | 5·10⁻² |
|---|---|---|---|---|---|---|---|
| p=1 | 46 | 1.000 | 0.983 | 0.951 | 0.848 | 0.721 | 0.447 |
| p=2 | 91 | 0.996 | 0.966 | 0.909 | 0.736 | 0.550 | 0.236 |
| p=3 | 137 | 0.995 | 0.951 | 0.869 | 0.640 | 0.421 | 0.127 |
| p=4 (submitted) | 182 | 0.993 | 0.934 | 0.830 | 0.555 | 0.320 | 0.068 |
| p=4, pruned tau=0.1 | 114 | 0.943 | 0.908 | 0.844 | 0.658 | 0.469 | 0.182 |
| p=4, pruned tau=0.25 | 69 | 0.891 | 0.871 | 0.832 | 0.710 | 0.568 | 0.300 |
| p=2, pruned tau=0.25 | 35 | 0.893 | 0.883 | 0.861 | 0.790 | 0.699 | 0.484 |

Mean lift on R + 8 unseen scenarios. Shallow circuits degrade far more slowly: at e₂ = 3·10⁻³, p = 1 keeps 0.951 against 0.830 for p = 4.

Routing on an IBM heavy-hex lattice (opt. level 3) roughly doubles the CNOT count: 187 → 360 at p = 4, measured with the previous trainer. Pruning couplings below 0.25·max|J| and refitting brings it to 47 at p = 2, with lift 0.836 at e₂ = 3·10⁻³, against 0.662 for the unpruned p = 4 circuit. On hardware, p and τ should therefore be chosen from a calibrated e₂.

**(c) Shot-based training** (honest negative). Estimating the objective from N shots per evaluation (SPSA, 300 evaluations) costs lift: VAL 0.844 at N = 1,024 and 0.814 at N = 256, vs 0.907 for exact training of the previous design. Exact coordinate updates rely on exact expectations, so on hardware NFT would need more shots per coordinate.

**(d) Parameter-transfer initialisation.** A single angle vector learned offline on 150 training scenarios lifts the COBYLA trainer from 0.907 to 0.940 on VAL. It is superseded by NFT, which needs no stored angles.

## 6. Limitations

- **Surrogate ceiling.** The quadratic surrogate cannot represent the min/max hinges exactly. In about 2 % of scenarios its minimizer is 5–9 % above C\*, and this bounds the raw lift. The hybrid repairs it.
- **Concentration.** The trained state is close to a product state: the mean single-qubit entropy is 0.04 bits. At 8 qubits, multi-angle QAOA concentrates almost all shots on the QUBO minimizer. The ZZ layers give no consistent gain at this size (ablation iii).
- **Coordinate-descent trap.** In 1 of 100 held-out scenarios (none on VAL), NFT converged to a near-basis state two bit-flips from the QUBO minimizer. That state ranks 13th by cost and is excluded from the elite set by the tie-break, so enrichment is 0 while lift stays 0.905. Single-angle updates cannot cross the two-flip barrier. We did not add restarts triggered by the known QUBO ground energy, because that would use classical knowledge of the answer.
- **Exact training.** Training uses exact expectations; extension (c) quantifies the shot-noise cost. Dense couplings dominate the noise budget.
- **No quantum advantage** is claimed at 8 qubits. Enumeration of 256 schedules is used only as a benchmark.
