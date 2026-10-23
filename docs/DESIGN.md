# Design and success rules, round 3: our own unlearning method (TPU). Committed before any result of this project was produced.

## Question
Rounds 1-2 showed that stochastic rounding lets a quantized model move, but the seven borrowed methods hurt the kept clients before they finish forgetting on MLP and PatchTST, even in full precision. Round 3 asks: **with our own unlearning method that protects the kept clients, does unlearning work (forgets correctly, kept clients intact) on a quantized model with no full-precision copy, on all 3 models and all 3 quantization techniques?**

## The method: TPU (Target-seeking, Protected-subset Unlearning) + stochastic rounding + controller
1. **Protected subset.** Per layer, only the 10% of weights whose squared gradient on forget windows is largest relative to their squared gradient on kept windows may change (weights that matter for the forgotten client and little for the others). If the target is not reached, the allowed share doubles every 200 steps (10%, 20%, 40%, 80%, 100%): protection is given up only as needed.
2. **Target-seeking loss.** `relu(1 - forget_loss / target)^2 + kept_loss / kept_loss_at_start`: the forget loss is pulled up to the target and the pull stops there (no blind ascent, no garbage labels).
3. **Normalised step.** Each step changes the weights by 0.3% of their total norm, along the masked gradient (no Adam).
4. **Stochastic rounding** to the integer grid after every step; scales fixed; no full-precision copy.
5. **Controller** (no retrained model): every 2 steps, on 256 forget windows and 256 kept-client *validation-period* windows, stop when the forget loss reaches the target, or the kept-client validation loss exceeds 1.10 x its start value, or after 1600 steps.
   Target = (75th percentile over the never-trained clients 11-20 of *model loss / previous-day-forecast loss*) x (previous-day-forecast loss on the forget windows).
Baseline in the same table: **MaxLoss** (best existing method of round 2) with the same stochastic rounding and controller (400 steps, its own fixed setting). Full precision (`fp`) of both methods is the reference. `nearest` rounding is the control.

## What was looked at while designing (stated openly)
On the design seed 0 (round-1/2 checkpoints) in full precision, I tried: Adam at several step sizes, normalised-gradient steps, subset sizes 2%-100%, more steps, and the growing subset. A masking bug (all weights allowed) was found and fixed during this. The 75th percentile (round 2 used the median) and the kept-client validation windows for the guard (round 2 used training windows, whose guard fired too early: kept-client damage measured on training data overstated the damage on held-out data) were changed after round 2's results. Settings were then frozen. **The gate is judged on fresh seeds only** (DLinear 5,6; MLP 4,5; PatchTST 2), trained after the freeze.

## Measurements (same as before)
forget score (0 = start model, 1 = retrained), retain ratio, MIA AUC. **Strict pass** = forget score 0.8-1.2 and retain ratio <= 1.10. Cell = architecture x quantizer (per-channel, per-tensor, 99th-percentile clipping) x bits (INT8, INT6, INT4) = 27 cells. A cell passes when all fresh seeds pass (2 seeds DLinear/MLP, 1 PatchTST).
Reference-free check as in round 2: `s_hat`, `retain_est`.

## Gate (decided now)
- **G1, the method works at all:** TPU in full precision passes strictly on each of the 3 architectures (all fresh seeds).
- **G2, works on quantized models:** TPU with stochastic rounding passes in at least 18 of the 27 cells, and in at least 5 of the 9 cells of each architecture.
- **G3, the rounding matters:** TPU with nearest rounding passes in at most 4 of the 18 cells at INT6 and INT4.
- **G4, better than what exists:** TPU with stochastic rounding passes in more cells than MaxLoss with stochastic rounding.
- **Verdict:** *established* if G1-G4 hold; *partly established* if G1 and G2 hold; *not established* otherwise. Failed cells are listed. Nothing is re-run to change this.
- **Reported, not gating:** median forget score, retain ratio, steps and seconds per run; reference-free check agreement; share of runs stopped by the target, by the guard, or by running out of steps; design-seed (exploratory) results.

## Limits stated up front
Simulated integers, biases/norm layers stay full precision, one dataset, one forgotten client, PatchTST fresh test = 1 seed, MaxLoss and TPU are the only methods in this round's table, settings were chosen on one design seed in full precision with the retrained model as judge (frozen before the fresh seeds), the target uses the forgotten client's windows and never-trained clients' windows (no retrained model).
