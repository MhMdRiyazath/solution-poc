# Replication on PatchTST seed 3 (rules in docs/ADDENDUM.md)

- R1 TPU in full precision passes: NO (score 0.64, retain 1.07)
- R2 TPU + stochastic passes 4 of 9 cells (need >= 6)
- R3 MaxLoss + stochastic passes 0 of 9 cells (need <= 2)
- TPU + nearest passes 1 of 9 cells

**Replication: NOT CONFIRMED**

TPU + stochastic by bit width (extra seed): INT8 2/3, INT6 0/3, INT4 2/3
With both PatchTST seeds (2 and 3), a cell passes only if both pass: 4 of 9 cells.

Per cell (extra seed), TPU + stochastic: forget score / retain ratio / steps

| tech    | 8                 | 6                 | 4                 |
|:--------|:------------------|:------------------|:------------------|
| channel | 1.1 / 1.06 / 288  | 1.24 / 1.07 / 296 | 0.53 / 1.05 / 180 |
| clip    | 1.24 / 1.05 / 296 | 1.48 / 1.06 / 346 | 0.87 / 1.04 / 374 |
| tensor  | 1.19 / 1.08 / 290 | 0.37 / 1.02 / 44  | 0.94 / 1.08 / 266 |