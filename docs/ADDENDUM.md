# Addendum: extra PatchTST seed (rules written before the run)

Why: the round-3 PatchTST result (TPU + stochastic rounding passes 8 of 9 cells) rests on one fresh seed. This adds one more seed, PatchTST seed 3, trained now; method settings, controller and quantizers are exactly those of round 3 (nothing re-tuned). The round-3 verdict in `results/verdict.md` is not changed.

Same 9 cells (per-channel, per-tensor, 99th-percentile clipping x INT8, INT6, INT4), same strict rule (forget score 0.8-1.2 and retain ratio <= 1.10), each cell judged on this single seed.

Replication counts as **confirmed** if all three hold: (R1) TPU in full precision passes; (R2) TPU + stochastic rounding passes in at least 6 of the 9 cells; (R3) MaxLoss + stochastic rounding passes in at most 2 of the 9 cells.
If R1 and R2 hold but not R3, or R2 holds with 4-5 cells, it is reported as **partly confirmed**; otherwise **not confirmed**. Reported either way. Also reported: pass counts per bit width and with both PatchTST seeds combined (a cell passes only if both seeds pass).
