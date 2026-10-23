# Verdict (rules from docs/DESIGN.md, applied as written, fresh seeds)

## G1: TPU works in full precision

- dlinear: FAIL
- mlp: FAIL
- patchtst: pass

**G1 DOES NOT HOLD**

## G2: TPU + stochastic rounding on quantized models

- dlinear: 3 of 9 cells pass (need >=5 of 9)
- mlp: 0 of 9 cells pass (need >=5 of 9)
- patchtst: 8 of 9 cells pass (need >=5 of 9)
- total 11 of 27 (need >=18 of 27)

**G2 DOES NOT HOLD**

## G3: nearest rounding as control
- TPU + nearest rounding passes in 0 of 18 cells at INT6/INT4 (need <=4)

**G3 HOLDS**

## G4: against the best existing method
- TPU stochastic 11 cells vs MaxLoss stochastic 0 cells

**G4 HOLDS**

## Verdict: **NOT ESTABLISHED**

Cells where TPU + stochastic did not pass: dlinear channel INT4 (score 2.04, retain 1.07, 729 steps); dlinear channel INT6 (score 1.55, retain 1.04, 1600 steps); dlinear clip INT4 (score 2.62, retain 1.09, 1195 steps); dlinear clip INT6 (score 1.45, retain 1.00, 1600 steps); dlinear tensor INT4 (score 2.31, retain 1.09, 702 steps); dlinear tensor INT6 (score 1.83, retain 1.05, 1490 steps); mlp channel INT4 (score 0.71, retain 1.10, 208 steps); mlp channel INT6 (score 0.75, retain 1.11, 165 steps); mlp channel INT8 (score 0.69, retain 1.10, 177 steps); mlp clip INT4 (score 0.81, retain 1.10, 207 steps); mlp clip INT6 (score 0.83, retain 1.11, 193 steps); mlp clip INT8 (score 0.93, retain 1.08, 198 steps); mlp tensor INT4 (score 0.51, retain 1.07, 142 steps); mlp tensor INT6 (score 0.73, retain 1.07, 202 steps); mlp tensor INT8 (score 0.69, retain 1.09, 177 steps); patchtst channel INT6 (score 0.64, retain 1.15, 72 steps)