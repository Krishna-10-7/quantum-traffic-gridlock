# Seed robustness - scenario R

C* = 72.0, C_rand = 151.250

| seed | E_Q | lift | enrichment | P(opt) | schedule cost | best sample | hybrid |
|---|---|---|---|---|---|---|---|
| 0 | 81.325 | 0.8823 | 15.56 | 0.424 | 72.0 | 72.0 | 72.0 |
| 1 | 81.393 | 0.8815 | 15.43 | 0.420 | 72.0 | 72.0 | 72.0 |
| 2 | 81.663 | 0.8781 | 15.35 | 0.404 | 72.0 | 72.0 | 72.0 |
| 3 | 82.050 | 0.8732 | 15.14 | 0.407 | 72.0 | 72.0 | 72.0 |
| 4 | 81.699 | 0.8776 | 15.35 | 0.413 | 72.0 | 72.0 | 72.0 |

**min lift (clipped) = 0.8732**, mean lift = 0.8785, mean enrichment = 15.36

| baseline | official cost |
|---|---|
| (1) best alternating | 116.0 |
| (2) greedy larger-queue | 92.5 |
| (3) exact enumeration C* | 72.0 |
| (4) uniform random expected | 151.250 |
| (5) QUBO exact minimizer (surrogate ceiling) | 72.0 |
| (6) QAOA raw samples, mean over 4096 shots (seed 0) | 81.325 |
| (6) QAOA decoded schedule (lowest QUBO energy sample) | 72.0 |
| (7) HYBRID: QAOA samples + 1-flip descent | 72.0 |
