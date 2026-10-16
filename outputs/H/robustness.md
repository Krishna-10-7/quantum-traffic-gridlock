# Seed robustness - scenario H

C* = 234.0, C_rand = 300.281

| seed | E_Q | lift | enrichment | P(opt) | schedule cost | best sample | hybrid |
|---|---|---|---|---|---|---|---|
| 0 | 236.691 | 0.9594 | 19.34 | 0.000 | 236.0 | 236.0 | 234.0 |
| 1 | 236.663 | 0.9598 | 19.39 | 0.000 | 236.0 | 236.0 | 234.0 |
| 2 | 236.728 | 0.9588 | 19.34 | 0.000 | 236.0 | 236.0 | 234.0 |
| 3 | 236.644 | 0.9601 | 19.37 | 0.000 | 236.0 | 236.0 | 234.0 |
| 4 | 236.595 | 0.9608 | 19.36 | 0.000 | 236.0 | 236.0 | 234.0 |

**min lift (clipped) = 0.9588**, mean lift = 0.9598, mean enrichment = 19.36

| baseline | official cost |
|---|---|
| (1) best alternating | 261.0 |
| (2) greedy larger-queue | 244.0 |
| (3) exact enumeration C* | 234.0 |
| (4) uniform random expected | 300.281 |
| (5) QUBO exact minimizer (surrogate ceiling) | 236.0 |
| (6) QAOA raw samples, mean over 4096 shots (seed 0) | 236.691 |
| (6) QAOA decoded schedule (lowest QUBO energy sample) | 236.0 |
| (7) HYBRID: QAOA samples + 1-flip descent | 234.0 |
