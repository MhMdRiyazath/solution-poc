# Results, round 3: our own method TPU (rules in docs/DESIGN.md were committed before the fresh-seed runs)

Tables: `results/RESULTS_TABLES.md`. Verdict as computed: `results/verdict.md`. Raw: `results/raw_*.csv`. Figure: `figures/r3.png`.
Fresh seeds: DLinear 5,6; MLP 4,5; PatchTST 2 (trained after the method was frozen). 3 quantizers x INT8/6/4 = 27 cells; a cell passes when every fresh seed passes the strict rule (forget score 0.8-1.2, retain ratio <= 1.10).

## Verdict as written in advance: NOT ESTABLISHED
- G1 (works in full precision on all 3 models): **fails** (PatchTST passes; MLP passes on 1 of 2 seeds; DLinear reaches 0.71 on both seeds, just under 0.8).
- G2 (stochastic rounding, at least 18 of 27 cells and at least 5 of 9 per model): **fails**, 11 of 27 cells: PatchTST 8 of 9, DLinear 3 of 9, MLP 0 of 9.
- G3 (nearest rounding passes at most 4 of 18 cells at INT6/INT4): holds, 0 of 18.
- G4 (more cells than MaxLoss with the same rounding and controller): holds, 11 against 0.

## What the data shows
1. **Our method works where the old ones did not, but not on every model yet.** Counting each fresh run on its own (after the fact, not the gate): TPU + stochastic rounding strictly passes in 19 of 45 quantized runs (PatchTST 8 of 9, DLinear 6 of 18, MLP 5 of 18); MaxLoss with the same rounding and controller passes 0 of 45, and nearest rounding 6 of 45 (all at INT8). The seven borrowed methods of round 2 passed 0 on MLP and PatchTST. Median forget score of TPU + stochastic: PatchTST 0.99 (retain 1.09), MLP 0.78 (1.10), against 0.65 (1.14) and 0.64 (1.19) for MaxLoss.
2. **All three quantizers work.** TPU + stochastic passes in 5 of 15 runs with per-channel, 7 of 15 with per-tensor and 7 of 15 with clipping quantization. By bit width: INT8 12 of 15, INT6 4 of 15, INT4 3 of 15: it degrades toward lower bit widths.
3. **Stochastic rounding is what makes it possible.** TPU with nearest rounding never moves the weights at INT6/INT4 (0 passes); it passes only at INT8 (6 runs).
4. **Where it fails, and why.**
   - *DLinear:* the model hardly memorizes (gap about 5%), so the forget score is hypersensitive; at INT6/INT4 stochastic rounding overshoots (median score 1.55, up to 2.6) with kept clients intact (retain 1.04), and in full precision it stops at 0.71 after 1600 steps. 
   - *MLP:* the result depends on the seed: full precision passes seed 5 (0.91, retain 1.05) and fails seed 4 (0.50, retain 1.14); the quantized runs follow the same split. Many MLP cells end at forget score 0.7-0.8 with retain 1.07-1.11, just outside the strict window.
   - *Both:* the stop target is an estimate; this is the main source of the near-misses.
5. **How runs ended (stochastic, all cells):** target reached 25 of 45 for TPU (MaxLoss 25 of 45), retain guard 9 (MaxLoss 20), ran out of steps 11 (all DLinear).
6. **The reference-free check is weak here:** for TPU stochastic runs it agrees with the retrained-model judge in 28 of 45 runs (62%); it predicted a pass in 22 runs and caught 12 of the 19 true passes.
7. **Cost:** median per run: DLinear 7.6 s (1600 steps), MLP 3.4 s (204 steps), PatchTST 19 s (202 steps) on one CPU thread.

## Honest reading
TPU, with stochastic rounding and no full-precision copy, makes unlearning work on a quantized time-series model on at least one model (PatchTST, all 3 quantizers at INT8/6/4 except one cell) and clearly better than the borrowed methods elsewhere (MLP, partly), but it does not meet the pre-registered bar of 18 of 27 cells. The two failure modes are known and specific: an unreliable stop target on models with a tiny or seed-dependent memorization gap (DLinear, MLP), and growing damage at INT4.

## Limits
One seed for PatchTST, two for the others; settings were selected on one design seed in full precision with the retrained model as judge and then frozen; only two methods in this table; simulated integers, biases/norms in full precision; one dataset and forgotten client. Items 1-2 and 4-7 are descriptive and partly after the fact.
