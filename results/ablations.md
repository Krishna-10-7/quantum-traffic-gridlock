# Ablations (scenario R + 40 random hidden-range scenarios, seed 0, 4096 shots)

| variant | R lift | R enrich. | mean lift | min lift | mean enrich. | mean P(opt) | regret<=5% |
|---|---|---|---|---|---|---|---|
| final (p=4, split cost, per-qubit mixer, Gibbs, adaptive QUBO) | 0.882 | 15.56 | 0.909 | 0.815 | 16.85 | 0.370 | 1.00 |
| depth p=1 | 0.727 | 12.34 | 0.717 | 0.543 | 10.69 | 0.138 | 1.00 |
| depth p=2 | 0.907 | 15.79 | 0.886 | 0.778 | 16.10 | 0.384 | 1.00 |
| depth p=3 | 0.885 | 15.20 | 0.907 | 0.817 | 16.72 | 0.374 | 1.00 |
| standard QAOA ansatz (1 gamma, 1 beta per layer) | 0.786 | 12.37 | 0.815 | 0.711 | 13.12 | 0.172 | 1.00 |
| per-qubit mixer only (1 gamma per layer) | 0.910 | 16.18 | 0.884 | 0.744 | 16.24 | 0.327 | 1.00 |
| objective <E> (standard) | 0.857 | 14.21 | 0.884 | 0.786 | 15.39 | 0.272 | 1.00 |
| objective CVaR alpha=0.25 | 0.777 | 11.93 | 0.793 | 0.588 | 13.53 | 0.233 | 1.00 |
| Gibbs eta=1 | 0.869 | 14.78 | 0.904 | 0.796 | 16.51 | 0.336 | 1.00 |
| Gibbs eta=4 | 0.883 | 14.45 | 0.910 | 0.817 | 17.10 | 0.420 | 1.00 |
| QUBO: uniform Walsh projection | 0.847 | 13.26 | 0.899 | 0.789 | 16.11 | 0.357 | 0.97 |
| QUBO: adaptive kappa=0.5 | 0.883 | 14.82 | 0.913 | 0.819 | 17.08 | 0.373 | 1.00 |
| QUBO: adaptive kappa=2 | 0.876 | 14.16 | 0.905 | 0.790 | 16.66 | 0.374 | 1.00 |
| init: random angles (3 probes) | 0.867 | 15.20 | 0.857 | 0.654 | 14.92 | 0.275 | 1.00 |
| init: near-identity | 0.828 | 16.00 | 0.889 | 0.765 | 15.91 | 0.325 | 1.00 |
| budget 100 evaluations | 0.778 | 11.66 | 0.809 | 0.689 | 12.62 | 0.162 | 1.00 |
| budget 200 evaluations | 0.847 | 14.50 | 0.872 | 0.773 | 15.39 | 0.269 | 1.00 |
