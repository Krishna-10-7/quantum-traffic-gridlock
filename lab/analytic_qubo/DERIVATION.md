# Analytic QUBO: second-order expansion of the queue recurrence at the Webster split

**Variables.** x = (A1..A4, B1..B4), with x = 1 for NS green. Approach k gets green g_{k,t} = x_{n,t} (NS) or 1 − x_{n,t} (EW).

**1. Exact per-approach dynamics.** Each approach follows the Lindley recursion q_t = (q_{t−1} + a_t + τ_t − c·g_t)^+ with q_0 = 0,
i.e. q_t = max_{0≤m≤t} Σ_{u=m+1..t} (a_u + τ_u − c·g_u). Its cost is D_k = Σ_t q_t + w_ov Σ_t (q_t − S)^+, and

C(x) = Σ_k D_k(x_{S_k}) + w_sw Σ_{n,t} (x_{n,t} + x_{n,t−1} − 2 x_{n,t} x_{n,t−1})   (the switching term is exact).

Locality fixes the window S_k. Each A approach sees A1..A4 and the unlinked B approach sees B1..B4. The linked B approach sees
B1..B4 and A1..A3, through the platoon τ_{t+1} = ⌊f·d^A_t + ½⌉ for t ≤ 3. A4's platoon leaves the model, so **every A4–B coupling is exactly 0**.

**2. Exact pairwise physics (Möbius expansion at the all-red state).** Let A_t = Σ_{u≤t} a_u and δ_s = min(c, A_s), with slots numbered 1..4.
- *Value of a green:* a green at s serves δ_s vehicles and saves (5 − s)·δ_s vehicle-slots.
- *Queue-emptying interference:* two greens s < t interact by +(5 − t)[min(c, A_t) − min(c, A_t − δ_s)] ≥ 0. This is 0 if the
  second green is still saturated, and positive if the first green already served vehicles the second would have served.
- *Spillback:* an A green at s ≤ 3 releases the platoon P_s = ⌊f·min(c, A^A_s)⌉. It costs (4 − s)·P_s at a red B. A B green at
  t > s serves it, which gives the green-wave coupling −(5 − t)[min(c, A^B_t + P_s) − min(c, A^B_t)] ≤ 0. Two A greens s < s′
  shrink the second platoon by (4 − s′)(⌊f·min(c, A^A_{s′} − δ_s)⌉ − P_{s′}) ≤ 0.
- *Saturation:* if every green of an approach meets a queue ≥ c, its delay is exactly linear in the greens.
- *Overflow hinge:* overflow terms are the same finite differences of (q − S)^+ on these explicit queue vectors.

All formulas match brute force with zero error (`closed_form.py`). Truncating at order 2 is exact while an approach has ≤ 2 greens,
but it misses the cascade when ≥ 3 greens compete for the same vehicles. On validation, only 38 % of scenarios reach regret ≤ 5 %.
The higher-order interference must therefore be *averaged into* the quadratic terms.

**3. Modelling rule.** Let D̄_k(p) = E_{x∼Bernoulli(p)} D_k(x) be the multilinear extension. Expand it to second order at the fluid
operating point p*:

E(x) = Σ_k [ D̄_k(p*) + Σ_i V_{ki}(x_i − p*_i) + Σ_{i<j} W_{kij}(x_i − p*_i)(x_j − p*_j) ] + switching,

- V_{ki} = E_{p*}[D_k | x_i = 1] − E_{p*}[D_k | x_i = 0] is the expected cost of turning slot i to NS;
- W_{kij} = E_{p*}[D_k(1,1) − D_k(1,0) − D_k(0,1) + D_k(0,0)] is the step-2 interaction, with the other slots' greens drawn from p*
  instead of all red.

The centred monomials are orthogonal under the product measure, so this is exactly the L²(Bernoulli(p*)) projection of D_k onto
quadratics, i.e. the best quadratic fit of the min/hinge nonlinearities near the operating point. p* = ½ gives the uniform Walsh
truncation, and a vertex gives step 2. We use the equal-saturation (Webster) split of cumulative demand:

p*_{n,t} = clip((N_{n,t} + ½)/(N_{n,t} + E_{n,t} + 1), 0.25, 0.75),

where N and E are the cumulative NS and EW arrivals, and the linked approach adds the expected platoon f·a^A_{t−1}.

**4. Cost and integrity.** The rule runs 176 single-approach recursions: 16 + 16 + 16 + (8 platoon vectors × 16). Every sum
factorises per approach (≤ 2^7 terms). It never simulates or assembles a joint 8-bit schedule cost and never uses the optimum.

The repo's adaptive projection can also be rebuilt *exactly* from these 176 local recursions (verified to 0 error). The naive local
count is 304 (16 + 16 + 16 + 256); the repo code runs 1,024. Its Gibbs weights do not factorise, though, so its normal equations
sum the assembled cost Σ_k D_k over all 2^8 joint schedules.

The rule yields 24 couplings instead of 28: 96 instead of 112 RZZ gates at p = 4.

**5. Evidence.** VAL = R + 60 unseen scenarios (seed 555), with QAOA settings unchanged and 4,096-shot metrics (seed 0).

| QUBO | regret ≤ 5 % | max regret | Pearson(E, C) | mean lift | min lift | enrichment | R lift (5 seeds, min) |
|---|---|---|---|---|---|---|---|
| adaptive projection (current) | 98.4 % | 5.2 % | 0.989 | **0.907** | **0.734** | **16.8** | 0.873 |
| **analytic, Webster point** | 98.4 % | 5.2 % | 0.990 | 0.902 | 0.718 | 16.6 | **0.884** |
| analytic, p* = ½ (uniform) | 96.7 % | 11.8 % | 0.992 | 0.895 | 0.717 | 16.2 | – |
| pairwise physics only (all-red) | 37.7 % | 73.5 % | 0.928 | 0.779 | 0.465 | 10.7 | – |

The analytic and adaptive Ising coefficients agree closely: Pearson 0.998 and relative L2 distance 8 % on VAL. The analytic
expansion therefore explains, term by term, what the projection QUBO encodes.
