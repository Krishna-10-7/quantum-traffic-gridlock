# Ablations (scenario R + 40 random hidden-range scenarios, seed 0, 4096 shots)

| variant | R lift | R enrich. | mean lift | min lift | mean enrich. | mean P(opt) | regret<=5% |
|---|---|---|---|---|---|---|---|
| final (Webster QUBO, p=4 field/per-qubit multi-angle QAOA, Gibbs, NFT trainer) | 0.993 | 19.51 | 0.982 | 0.867 | 19.20 | 0.729 | 1.00 |
| depth p=1 | 1.000 | 19.69 | 0.987 | 0.872 | 19.44 | 0.650 | 1.00 |
| depth p=2 | 0.997 | 19.61 | 0.987 | 0.848 | 19.26 | 0.785 | 1.00 |
| depth p=3 | 0.996 | 19.59 | 0.983 | 0.861 | 19.09 | 0.699 | 1.00 |
| no ZZ couplings in circuit (product-state ansatz) | 1.000 | 19.69 | 0.985 | 0.932 | 19.69 | 0.563 | 1.00 |
| previous submission: split cost, COBYLA trainer | 0.888 | 15.13 | 0.901 | 0.795 | 16.35 | 0.349 | 1.00 |
| standard QAOA ansatz, COBYLA | 0.776 | 11.68 | 0.815 | 0.690 | 13.20 | 0.177 | 1.00 |
| field ansatz, COBYLA trainer | 0.834 | 14.14 | 0.870 | 0.745 | 15.28 | 0.250 | 1.00 |
| objective <E> (standard) | 0.995 | 19.54 | 0.984 | 0.864 | 19.16 | 0.698 | 1.00 |
| objective CVaR alpha=0.25 (COBYLA) | 0.807 | 12.87 | 0.799 | 0.671 | 13.35 | 0.221 | 1.00 |
| Gibbs eta=1 | 0.994 | 19.52 | 0.983 | 0.867 | 19.23 | 0.711 | 1.00 |
| Gibbs eta=4 | 0.993 | 19.51 | 0.982 | 0.868 | 19.21 | 0.722 | 1.00 |
| QUBO: adaptive projection of full cost table | 0.992 | 19.50 | 0.987 | 0.948 | 19.46 | 0.765 | 1.00 |
| QUBO: uniform Walsh projection | 0.995 | 19.57 | 0.979 | 0.839 | 19.22 | 0.712 | 0.97 |
| QUBO: pairwise physics only (all-red expansion) | 0.904 | 19.64 | 0.867 | 0.408 | 14.42 | 0.303 | 0.53 |
| init: random angles (3 probes) | 0.997 | 19.63 | 0.974 | 0.848 | 19.02 | 0.708 | 1.00 |
| init: near-identity | 1.000 | 19.69 | 0.989 | 0.849 | 19.65 | 0.710 | 1.00 |
| budget 100 evaluations | 0.921 | 17.28 | 0.928 | 0.836 | 17.43 | 0.626 | 1.00 |
| budget 200 evaluations | 0.988 | 19.30 | 0.977 | 0.861 | 18.97 | 0.714 | 1.00 |
