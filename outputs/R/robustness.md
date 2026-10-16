# Seed robustness - scenario R

C* = 72.0, C_rand = 151.250

| seed | E_Q | lift | enrichment | P(opt) | schedule cost | best sample | hybrid |
|---|---|---|---|---|---|---|---|
| 0 | 72.523 | 0.9934 | 19.51 | 0.985 | 72.0 | 72.0 | 72.0 |
| 1 | 72.470 | 0.9941 | 19.53 | 0.987 | 72.0 | 72.0 | 72.0 |
| 2 | 72.470 | 0.9941 | 19.51 | 0.985 | 72.0 | 72.0 | 72.0 |
| 3 | 72.482 | 0.9939 | 19.52 | 0.986 | 72.0 | 72.0 | 72.0 |
| 4 | 72.408 | 0.9948 | 19.54 | 0.985 | 72.0 | 72.0 | 72.0 |

**min lift (clipped) = 0.9934**, mean lift = 0.9941, mean enrichment = 19.52

| baseline | official cost |
|---|---|
| (1) best alternating | 116.0 |
| (2) greedy larger-queue | 92.5 |
| (3) exact enumeration C* | 72.0 |
| (4) uniform random expected | 151.250 |
| (5) QUBO exact minimizer (surrogate ceiling) | 72.0 |
| (6) QAOA raw samples, mean over 4096 shots (seed 0) | 72.523 |
| (6) QAOA decoded schedule (lowest QUBO energy sample) | 72.0 |
| (7) HYBRID: QAOA samples + 1-flip descent | 72.0 |
