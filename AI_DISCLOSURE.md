# AI-use disclosure (outside the 3-page report limit)

**Tool used.** Claude (Anthropic; Claude Code agent, model Claude Opus 5.5), during the hackathon window.

**What the AI did.**
- Read the problem statement, `scorer.py` and `scenario_R.json`, and proposed the modelling approach (per-approach
  decomposition, quadratic Walsh projection, adaptive measure) and the QAOA design (multi-angle mixer, split cost angles,
  Gibbs objective, linear-ramp initialisation, two-stage budgeted COBYLA).
- Wrote the code in this repository (`run.py`, `traffic_model.py`, `qubo.py`, `qaoa.py`, `circuit.py`, `baselines.py`,
  `experiments/`, `tests/`) and drafted `README.md` and `REPORT.md`.
- Ran the exploratory sweeps used to choose hyper-parameters. Hyper-parameters were selected on randomly generated
  scenarios (generator seeds 1, 7, 11) and then **frozen**. The reported generalization (seed 2026) and ablation
  (seed 99) sets were not used for selection. Scenario R was never used to tune anything beyond being one point among many.

**How results were independently verified.**
- `scorer.py` is the unmodified official file. All reported costs come from `scorer.evaluate`. The four published
  verification vectors are re-checked in `tests/test_pipeline.py`.
- The per-approach decomposition is checked against the scorer on 1,240 random (scenario, schedule) pairs.
- The QUBO↔Ising conversion, the Walsh-orthogonality of the uniform projection, and the identity between the numpy trainer
  and the Qiskit circuit (max probability difference ≈ 1e-15) are asserted on every run.
- OpenQASM 3 round trip and counts-vs-circuit consistency are tested. Counts come from Qiskit's seeded `StatevectorSampler`
  on the exported circuit.
- Budget compliance (≤ 300 evaluations, p ≤ 4, ≤ 16 qubits, exactly 4,096 shots) is asserted at run time.
- One AI-introduced bug was caught by these checks during development: an inverted mixer sign that made the
  annealing initialisation start on the wrong side. It was fixed before any reported run.

**What the AI did not do.** It did not access hidden scenarios. It did not hard-code any schedule or scenario-specific value,
and it did not use the official cost of any schedule inside the QAOA objective. The decoded raw-QAOA schedule is
chosen by QUBO energy only.

**Team responsibility.** The team reviewed the derivation, reran every command in `README.md`, and takes responsibility
for the submission. *(Team: please edit this paragraph to describe your own contributions accurately.)*
