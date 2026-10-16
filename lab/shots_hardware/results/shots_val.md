# Shot-based training on VAL (R + 60 scenarios, generator seed 555)

Final metrics: 4096 shots of the final bound circuit, sampling seed 0 (as run.solve). R robustness: seeds 0-4 (seed drives training shots AND final shots).

| training | VAL mean lift | VAL min lift | VAL mean enrich. | VAL min enrich. | P(opt) | R lift (seed 0) | R min lift (5 seeds) | R mean lift | R mean enrich. |
|---|---|---|---|---|---|---|---|---|---|
| exact (current final config) | 0.907 | 0.734 | 16.79 | 10.82 | 0.343 | 0.882 | 0.873 | 0.879 | 15.36 |
| naive COBYLA, N=256 | 0.669 | 0.491 | 7.65 | 4.03 | 0.058 | 0.698 | 0.675 | 0.700 | 8.60 |
| SPSA, N=128 | 0.784 | 0.358 | 12.52 | 5.20 | 0.164 | 0.739 | 0.739 | 0.777 | 12.25 |
| SPSA, N=256 | 0.814 | 0.594 | 13.19 | 5.50 | 0.178 | 0.792 | 0.787 | 0.815 | 13.34 |
| SPSA, N=1024 | 0.844 | 0.636 | 14.13 | 7.43 | 0.204 | 0.764 | 0.762 | 0.804 | 12.76 |
| SPSA, exact objective (N=1e8) | 0.851 | 0.700 | 14.58 | 6.57 | 0.215 | 0.835 | 0.800 | 0.821 | 12.87 |

Surrogate regret <= 5% on 98.4% of VAL (same QUBO for all rows).
