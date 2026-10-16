# Quantum Traffic Gridlock — QUBO + QAOA signal coordination

IBM Qiskit Fall Fest 2026 · QC@IITI. A scenario-independent pipeline that turns a traffic scenario into a
traffic-derived QUBO, trains a budgeted QAOA (p = 4, 8 qubits, ≤ 300 objective evaluations), samples exactly
4,096 shots from the final bound circuit and reports every required baseline using the **official scorer**.

## Quick start

```bash
pip install -r requirements.txt          # core needs only qiskit, numpy, scipy
python run.py --scenario scenarios/scenario_R.json --seed 0 --out outputs/R/seed0
```

Runtime ≈ 1 s per seed on a laptop CPU (limit: 10 min), < 300 MB RAM, no network, deterministic per seed.

Output files in `--out`:

| file | content |
|---|---|
| `qubo.json` | QUBO matrix `Q_upper_triangular` + `offset` (E = offset + xᵀQx), Ising `h`, `J`, `const`; variable→schedule-bit and variable→qubit maps; `auxiliary_variables: []` (none used) |
| `circuit.qasm` | final bound-parameter QAOA circuit, OpenQASM 3 |
| `counts.json` | exactly 4,096 raw shots of that circuit, bitstring → count |
| `schedule.json` | decoded schedule + bit order documentation, alternatives (most frequent, QUBO minimizer, labelled hybrid) |
| `metrics.json` | all required baselines (1)–(6) + labelled hybrid, lift, enrichment, surrogate regret, circuit resources, runtime |
| `training.json` | every training objective value, final parameters, physical angles per layer |

**Bit order (everywhere):** `A_slot1 A_slot2 A_slot3 A_slot4 B_slot1 B_slot2 B_slot3 B_slot4`, `1` = NS green, `0` = EW green —
identical to `scorer.py`. Schedule bit *i* is placed on qubit *7−i* and measured into clbit *7−i*, so the Qiskit count keys
(`c7…c0`) **are** schedule strings: key `11101010` means A = 1110, B = 1010. Check any key with
`python scorer.py --scenario <file> --schedule <key>`.

## Reproduce everything

```bash
python tests/test_pipeline.py                                               # 9 verification tests
python experiments/benchmark.py seeds --scenario scenarios/scenario_R.json --out outputs/R   # 5 seeds -> robustness.md
python experiments/benchmark.py seeds --scenario scenarios/scenario_H.json --out outputs/H
python experiments/benchmark.py generalization --n 100 --gen-seed 2026      # results/generalization.json
python experiments/benchmark.py ablations --n 40 --gen-seed 99              # results/ablations.md
python experiments/noise.py                                                 # results/noise.md (extension)
python experiments/figures.py                                               # figures/
```

## Headline results

| | Scenario R (5 seeds) | Scenario H (5 seeds) | 100 unseen random scenarios (hidden ranges) |
|---|---|---|---|
| QAOA raw lift | **0.993 min / 0.994 mean** | **0.959 min / 0.960 mean** | 0.979 mean, 0.819 min |
| Elite-13 enrichment (random = 1, max 19.69) | **19.52** | **19.36** | 19.28 |
| Shots on the exact optimum | 98.5 % | 0 % (shots on 2nd-best, cost 236) | 60 % mean |
| Decoded schedule | `11101010`, cost **72.0 = C\*** | `10010110`, 236.0 (C\* = 234.0) | optimal in 62 %, mean excess 0.61 % |
| Surrogate regret (QUBO minimizer) | 0 % | 0.85 % | ≤ 5 % in 97 % of scenarios |

Scenario H baselines: alternating 261.0 · greedy 244.0 · uniform random 300.28 · exact C\* 234.0 · QUBO minimizer 236.0 · hybrid 234.0.
Scenario R baselines: alternating 116.0 · greedy 92.5 · uniform random 151.25 · exact C\* 72.0 · QUBO minimizer 72.0.

**Method in one line:** closed-form QUBO, i.e. the second-order expansion of each approach's exact queue recursion at the
Webster split (176 single-approach recursions, no 256-schedule cost table), plus multi-angle QAOA (p = 4, 68 angles)
trained with exact NFT/Rotosolve coordinate updates on a Gibbs objective, within 300 evaluations.

## Layout

```
run.py               required entry point
traffic_model.py     per-approach decomposition of the official dynamics, scenario generator
analytic_qubo.py     DEFAULT modelling rule: Webster-point second-order expansion of each approach's queue cost
qubo.py              QUBO dispatch, Ising conversion, reference full-table projections (ablations)
qaoa.py              multi-angle ansatz, numpy statevector, Gibbs/CVaR/energy objectives, COBYLA trainer (ablation)
qaoa_nft.py          DEFAULT trainer: budgeted exact sequential minimal optimisation (NFT/Rotosolve)
circuit.py           Qiskit circuit, OpenQASM 3 export, StatevectorSampler (seeded) for the 4,096 shots
baselines.py         alternating, greedy, exact, random, QUBO-ceiling, metrics, labelled hybrid
scorer.py            official evaluator (unmodified copy)
experiments/         seeds, generalization, ablations, noise extension, figures, PDF build
lab/                 research tracks with their own scripts + RESULTS.md: analytic_qubo, optimizer_ansatz (NFT),
                     transfer_init, chain_extension (A->B->C, 12 qubits), shots_hardware (shot training, heavy-hex)
tests/               independent verification
REPORT.md / REPORT.pdf   technical report (≤ 3 pages)      AI_DISCLOSURE.pdf  disclosure (1 page)
```
