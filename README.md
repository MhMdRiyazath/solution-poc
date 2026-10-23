# qunlearn-r3: our own unlearning method (TPU) with stochastic rounding for quantized time-series models

Round 3 of the PoC. Same data, models and quantizers as rounds 1-2; new method TPU (protected-subset, target-seeking), new fresh seeds, all 3 quantizers.
Read in order: `docs/DESIGN.md` (rules fixed first) -> `results/verdict.md` -> `docs/RESULTS.md` -> `results/RESULTS_TABLES.md`.
Addendum: `docs/ADDENDUM.md` and `results/replication.md` (extra PatchTST seed; the earlier PatchTST result did not replicate).
Steps: `01_download`, `02_preprocess`, `03_train --arch A`, `04_quantize_unlearn --arch A --seed S`, `05_report`; `run_all.sh` is the run used.
