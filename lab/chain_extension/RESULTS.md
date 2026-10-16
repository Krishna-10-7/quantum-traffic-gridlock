# Three-intersection chain extension (A -> B -> C, 12 qubits)

Innovation and extensions track. The point is not scale. The point is to check that the SAME
scenario-independent modelling rule (weighted-L2 quadratic projection with the adaptive measure) and the SAME
frozen QAOA design (`run.DEFAULTS`) carry over unchanged to a larger network, and to measure what changes.
No file outside this directory was modified.

## 1. Definition

Intersections A, B, C; slots t = 1..4; 12 bits x = (A1..A4, B1..B4, C1..C4), 1 = NS green, 0 = EW green
(the same order as Qiskit keys; schedule bit i is on qubit 11-i). Two directed links: l1 = A->B on phase
phi1 with fraction f1, and l2 = B->C on phase phi2 with fraction f2. Shared parameters are capacity c, storage S,
w_ov and w_sw. The slot semantics are copied line by line from `scorer.py` (`chain_scorer.py`):

```
for t = 1..4, for n in (A, B, C), for phase p in (NS, EW):
    inflow_n,p(t) = a_n,p(t) + [p = phase of link into n] * pend_n(t)
    avail = q_n,p + inflow ;  dep = [p green at (n,t)] * min(avail, c) ;  q_n,p = avail - dep
    delay += q_n,p ;  overflow += max(0, q_n,p - S)
    if p = phase of link out of n and t < 4:  pend_{n+1}(t+1) += floor(f * dep + 1/2)   (exact decimal half-up)
switches = sum_n sum_{t=2..4} [x_n,t != x_n,t-1]
C3(x) = delay + w_ov * overflow + w_sw * switches
```

If phi1 = phi2 (an "aligned corridor"), vehicles that came from A can be passed on to C.

**Reduction (proved by construction, tested exhaustively).** If C has zero arrivals and f2 = 0, then for all
4096 schedules C3(xA, xB, xC) = C_official(xA xB) + w_sw * sw(xC). In particular,
C3(xA, xB, 0000) = C3(xA, xB, 1111) = `scorer.evaluate` exactly. The B->C link reduces the same way when A is
empty and f1 = 0. `test_reduction_to_official_AB` checks this on scenario R plus 30 official random scenarios, for
all 4096 schedules each, against the unmodified `scorer.py`. It also reproduces the four verification vectors in the
problem statement (116.0, 146.0, 237.0, 92.5) through the chain scorer.

**Light-cone theorem.** A transfer leaves node n in slot t and arrives at node n+1 in slot t+1, and departures after
slot 4 leave the model. So approach (n, p), reached through m consecutive links on the same phase p, depends only on
the upstream node n-m's bits in slots 1..4-m. Exact decomposition:
C3(x) = sum_k D_k(x_{S_k}) + w_sw * sum XOR(x_n,t, x_n,t-1), with six approach terms D_k = delay_k + w_ov * overflow_k:

| approach | scope S_k | size |
|---|---|---|
| A, either phase; B or C on an unlinked phase | own 4 bits | 4 |
| B on phi1 | B1..B4, A1..A3 | 7 |
| C on phi2, phi2 != phi1 | C1..C4, B1..B3 | 7 |
| C on phi2 = phi1 (aligned) | C1..C4, B1..B3, A1..A2 | 9 |

For a corridor of any length with T slots, |S_k| <= T(T+1)/2 = 10. `test_light_cone_scopes` checks that every
D_k is constant outside its scope (all 4096 inputs, 61 scenarios) and that the scopes are tight: every listed bit
matters in some scenario. `corridor.py` checks the general L-node statement for L = 4 (16 bits, 65,536 schedules,
12 corridors) and that the L = 3 corridor code equals the chain model.

## 2. QUBO: the same rule, and what it costs

E = argmin over quadratic g of sum_x mu(x) (C3(x) - g(x))^2, with g(x) = offset + sum_{i<=j} Q_ij x_i x_j on
{0,1}^12. The measure is mu_kappa(x) ∝ exp(-(E(x) - min E) / (kappa * std E)), solved as an 8-iteration fixed point
with kappa = 1, exactly as in `qubo.py`. This gives 12 variables, no auxiliary variables, and at most 66 couplings.
The weights depend only on the surrogate's own energies, never on the optimum or a ranking of C3.

* **Global construction** (used in the pipeline): it evaluates the vectorised recurrence on all 4096 inputs,
  about 3-8 ms, plus a weighted least-squares fit, about 0.2-0.27 s. That is 16x the 8-bit case and is still
  negligible here.
* **Approach-wise (local) construction, uniform measure** (`build_qubo_local_uniform`): the uniform measure is a
  product measure, so the projection of each D_k only involves the bits in S_k. It needs sum_k 2^|S_k| simulations:
  **704** for an aligned chain and **320** otherwise, versus 4096. It is tested to equal the global uniform
  projection exactly. Couplings appear only on light-cone pairs: 50 of 66 when aligned, 42 when not. For aligned
  corridors the count grows linearly: 1744 at L = 4 versus 2^16, and 5904 at L = 8 versus 2^32 (`corridor.py`).
* **Limitation (honest).** The adaptive measure is not a product measure, so the local construction is NOT exact for
  it. At 12 bits this does not matter. At larger L it would need a sampled or local approximation, which was not
  implemented. The 2^n statevector emulation of QAOA hits the 16-qubit cap well before the QUBO construction cost
  matters.

## 3. QAOA: the same design

`chain_qaoa.py` is `qaoa.py` with n read from the Hamiltonian instead of the constant 8. The ansatz, the objectives
(`qaoa.make_objective` is imported unchanged; it uses QUBO energies only), and the two-stage budgeted trainer are the
same: 3 linear-ramp probes, COBYLA on the standard 2p-parameter ansatz until evaluation 40, then embed into the rich
ansatz (split gamma, per-qubit beta) and COBYLA to 300. Every evaluation counts. `test_nqubit_trainer_reduces_to_repo_trainer`
shows that at n = 8 this matches the repo `qaoa.Ansatz` and `qaoa.train` bit for bit, with identical 300-evaluation
histories.

`chain_circuit.py` builds the 12-qubit Qiskit circuit: H, then RZ and RZZ, then RX(-2b) per layer. The circuit
matches the trainer's statevector to within 1e-10, and the bit-order test passes. Sampling uses exactly 4096 shots
from StatevectorSampler (seed = run seed). Final config: p = 4, Gibbs eta = 2, adaptive kappa = 1. Each run uses
**300 evaluations, 12 qubits, 56 parameters** (4 x (2 + 12)), mean 246 RZZ gates and depth 65.
`chain_run.py --scenario --seed --out` writes `qubo.json`, `circuit.qasm`, `counts.json`, `schedule.json`,
`metrics.json` and `training.json`.

## 4. Protocol

* **Scenarios** (`chain_model.random_chain_scenario`): official hidden ranges applied to all three nodes and both
  links. Arrivals 0-8, capacity 4-6, storage 5-8, w_ov in {2..6 step 0.5}, w_sw in {0.25..1.5 step 0.25}, link
  phases NS/EW, fractions {0.4..0.8}. Same filter as the repo: C_rand - C* > 5.
* **Tuning set:** generator seed 31, 12 scenarios. It was used only for design checks, and no setting was changed:
  the 2-node `run.DEFAULTS` is used as is.
* **Report set:** R3 + 20 scenarios from generator seed 1001, run seeds 0-4 (105 runs). R3 is scenario R with C
  copying B's arrivals and a B->C link identical to R's A->B link. Nothing was selected on this set.
* **Metric:** 4096-shot sampled metrics with the chain scorer's costs. lift = (C_rand - E_Q) / (C_rand - C*).
  Elite set = the 208 = 4096 x 13/256 lowest-cost schedules (ties by ascending bitstring), so random enrichment is 1
  and the maximum is 19.69, the same scale as the official definition. Regret is the C3 cost of the QUBO exact
  minimiser relative to C*.

## 5. Headline results (report set, `results/report.json`)

| | n runs | mean lift | min lift | mean enrichment | min enrichment | P(opt) | regret <= 5 % | mean regret | max regret |
|---|---|---|---|---|---|---|---|---|---|
| 20 random chains x 5 seeds | 100 | **0.871** | 0.753 | **17.80** | 12.34 | 0.116 | 18/20 (0.90) | 1.15 % | 8.3 % |
| R3 x 5 seeds | 5 | 0.812 | 0.810 | 17.66 | 17.60 | 0.078 | 1/1 | 0 % | 0 % |
| all | 105 | 0.869 | 0.753 | 17.80 | 12.34 | 0.114 | 0.90 | 1.09 % | 8.3 % |

* Robustness (min lift over 5 seeds, per scenario): mean 0.866, worst 0.753. Seed-to-seed spread is tiny. The mean
  lift per seed is 0.870-0.873, and the exact-distribution lift is 0.869, so shot noise is negligible.
* The surrogate is good: Pearson(E, C3) is 0.991 on average (min 0.981). The two scenarios over 5 % regret are
  chain1001_4 (8.3 %) and chain1001_6 (7.3 %).
* Runtime per seed is **6.0 s mean, 7.8 s max**, including benchmarks, with 2 workers on a shared CPU. That is far
  below 10 minutes. The optimum C* appeared among the 4096 raw shots in every one of the 105 runs (min P(opt) = 0.0022).

**Baselines** (20 random chains; gap closed = (C_rand - cost) / (C_rand - C*); the mean relative gap to C* is in
brackets):

| method | gap closed | optimal | mean gap to C* |
|---|---|---|---|
| best alternating (8 schedules) | 0.666 | 0/20 | 24.2 % |
| greedy larger-queue (chain dynamics) | 0.881 | 0/20 | 5.9 % |
| sequential: optimise (A,B) alone, then C | 0.913 | 7/20 | 3.5 % |
| raw QAOA sample mean E_Q | 0.871 | - | - |
| raw QAOA decode (lowest QUBO energy among shots) = QUBO exact minimiser | 0.987 | 57 % of 105 runs | 1.15 % |
| HYBRID (labelled): QAOA top-8 samples + 1-flip descent on C3 | 0.9999 | 95 % of 105 runs | 0.01 % |
| exact enumeration | 1 | 20/20 | 0 |

The sequential baseline, which ignores C when choosing A and B, misses the optimum in 13/20 scenarios. Joint
12-bit coordination matters. Aligned and non-aligned chains give the same lift (0.870 vs 0.873). Aligned chains have
lower regret (0.5 % vs 1.8 %).

## 6. Ablations (one change each, report set, seed 0, 21 scenarios; `results/ablations_report_set.json`)

| variant | lift | min lift | enrichment | P(opt) | regret <= 5 % | RZZ |
|---|---|---|---|---|---|---|
| **final (transferred unchanged)** | **0.870** | 0.761 | 17.83 | 0.115 | 0.90 | 246 |
| standard ansatz (single gamma, single beta) | 0.798 | 0.699 | 16.02 | 0.046 | 0.90 | 246 |
| split gamma, single beta | 0.787 | 0.701 | 15.40 | 0.042 | 0.90 | 246 |
| p = 1 / 2 / 3 | 0.703 / 0.853 / 0.867 | 0.482 / 0.723 / 0.729 | 12.57 / 17.13 / 17.72 | | 0.90 | 62 / 123 / 185 |
| uniform measure (local construction) | 0.864 | 0.709 | 17.60 | 0.112 | **0.81** | 162 |
| adaptive, light-cone-sparse features | 0.871 | 0.761 | 17.86 | 0.116 | 0.90 | **178** |
| kappa = 0.5 | 0.874 | 0.751 | 17.95 | 0.123 | **1.00** | 246 |
| random initialisation (instead of ramp) | 0.816 | 0.660 | 16.34 | 0.086 | 0.90 | 246 |

The tuning set (seed 31, 12 scenarios, `results/tune_seed31.json`) gives the same overall picture. The final design
gets 0.877, the standard ansatz 0.806, p = 2 / 3 give 0.875 / 0.878, uniform measure 0.850, random initialisation
0.828, and kappa = 0.5 / 2 give 0.882 / 0.870. Also: eta = 1 / 4 gives 0.867 / 0.874, the plain <E> objective
gives 0.859, and stage-1 budgets of 20 / 80 give 0.875 / 0.873.

* The per-qubit mixer is the component that carries over most strongly: about +0.07 lift over the standard ansatz,
  similar to 8 qubits.
* The **light-cone-sparse** adaptive fit is free: same lift and regret with 28 % fewer RZZ gates. It uses the
  topology's physical coupling graph.
* kappa = 0.5 is slightly better on both the tuning and the report sets, and brings regret within 5 % on 21/21.
  I report it as an observation and did NOT adopt it. The frozen 2-node design was kept, so the transfer claim stays
  clean.

## 7. What changes from 8 to 12 qubits

* **Same traffic, two sizes** (`diagnose.py subproblem`, tuning set): the repo's unmodified `run.solve` on the A,B
  sub-network (8 qubits) gives lift 0.914, enrichment 16.86 and P(opt) 0.273. The 12-qubit chain gives lift 0.877,
  enrichment 18.46 and P(opt) 0.110. Lift drops by about 0.04 and P(opt) by about 2.5x, because the optimum's basin
  is spread over 16x more states. Enrichment rises because the 208-state elite set is a much larger target.
* **The budget is now the binding constraint** (`diagnose.py budget`, exact-distribution lift, tuning set). The rich
  ansatz gives 0.784, 0.845 and **0.876** at 100, 200 and 300 evaluations. Diagnostic-only budgets above the cap give
  0.915 at 600 and 0.940 at 1000. The standard ansatz saturates at 0.807 / 0.817 / 0.818 (300 / 600 / 1000). The
  56-parameter rich ansatz is under-trained at 300 evaluations on 12 qubits, which is not the case at 8 qubits.
  Larger networks would need parameter sharing, for example per-node beta (3 x p parameters) instead of per-qubit
  beta. This was not tested.
* QUBO construction is 16x more expensive (4096 simulations). It remains negligible here, and the local uniform
  construction shows how to avoid it (section 2).

## 8. Rule compliance

12 qubits (cap 16); p = 4; every training-objective evaluation counts, probes included (300 per run, asserted);
exactly 4096 final shots of the bound circuit (asserted); deterministic per seed (two identical runs give identical
counts, tested); CPU only. The QUBO comes from the documented projection rule on the simulated cost table. The
optimum is never located or encoded, and nothing is scenario-specific or hardcoded. The training objective uses QUBO
energies only. The chain scorer is used only for reporting. The hybrid result is labelled separately and is never
mixed into raw counts.

## 9. Reproduce (from the repo root)

```
python lab/chain_extension/test_chain.py                 # 9 tests, all pass, ~20-30 s (reduction to scorer.py included)
python lab/chain_extension/corridor.py                   # L = 4 light-cone theorem and local-construction cost, ~3 s
python lab/chain_extension/benchmark_chain.py report     # headline table (section 5), ~5.5 min with 2 workers
python lab/chain_extension/benchmark_chain.py ablate --extra "p=1" "p=3" "kappa=0.5" "init random"   # section 6
python lab/chain_extension/benchmark_chain.py tune       # tuning-set design checks
python lab/chain_extension/diagnose.py subproblem        # 8 vs 12 qubits on the same traffic
python lab/chain_extension/diagnose.py budget            # lift vs evaluation budget (600/1000 diagnostic only)
python lab/chain_extension/chain_run.py --scenario lab/chain_extension/scenarios/R3.json --seed 0 --out <dir>
```

The full log of the last run is `results_main.log`. The scenarios used are in `scenarios/`.
