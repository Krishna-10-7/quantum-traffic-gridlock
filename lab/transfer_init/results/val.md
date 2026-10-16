# VAL = scenario R + random_scenarios(60, seed=555); 4096 shots, sampler seed 0

| variant | R lift | R enrich. | mean lift (60) | min lift (60) | mean enrich. (60) | min enrich. | mean P(opt) | exact mean lift | evals max |
|---|---|---|---|---|---|---|---|---|---|
| current (run.solve, repo defaults) | 0.882 | 15.56 | 0.9074 | 0.734 | 16.81 | 10.82 | 0.342 | 0.9078 | 300 |
| candidate: transfer+ramp (split/qubit p=4, eta=2) | 0.939 | 18.00 | 0.9396 | 0.773 | 18.08 | 12.58 | 0.445 | 0.9403 | 300 |
| init ablation: ramp (= run.solve, re-implemented) | 0.882 | 15.56 | 0.9074 | 0.734 | 16.81 | 10.82 | 0.342 | 0.9078 | 300 |
| init ablation: random (3 probes, seed 0) | 0.867 | 15.20 | 0.8597 | 0.611 | 15.10 | 7.00 | 0.266 | 0.8600 | 300 |
| init ablation: transfer only | 0.939 | 18.00 | 0.9404 | 0.776 | 18.10 | 12.63 | 0.446 | 0.9407 | 300 |
