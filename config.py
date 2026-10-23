"""All settings in one place."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RAW_FILE, INTERIM, PROCESSED, CKPT = ROOT / "data/raw/electricity.txt.gz", ROOT / "data/interim", ROOT / "data/processed", ROOT / "checkpoints"
RES, FIG = ROOT / "results", ROOT / "figures"
for p in (INTERIM, PROCESSED, CKPT, RES, FIG): p.mkdir(parents=True, exist_ok=True)

# ---- forecasting task: hourly electricity, look-back 1 week, forecast next 24 h ----
L, H, PATCH, STRIDE = 168, 24, 16, 8
TRAIN_CLIENTS, FORGET, CONTROL = list(range(10)), 9, list(range(10, 20))   # client 10 (index 9) is the one to forget; clients 11-20 never trained on
TRAIN_END, VAL_END, WINDOW_STRIDE = 0.70, 0.80, 4
EPOCHS, BATCH, LR = 20, 128, 1e-3
ARCHS = ["dlinear", "mlp", "patchtst"]
SEEDS = {"dlinear": [5, 6], "mlp": [4, 5], "patchtst": [2, 3]}      # fresh seeds, trained after the method was designed (seed 0, the design seed, can be run as exploratory)
FRESH = {"dlinear": [5, 6], "mlp": [4, 5], "patchtst": [2]}
REPLICATION = {"patchtst": [3]}        # extra seed added after the round-3 verdict, to check the PatchTST result (docs/ADDENDUM.md)      # the gate is judged on these only

# ---- quantization: post-training, weights of every Linear layer, signed integers ----
BITS = [8, 6, 4, 3, 2]
QUANT = {"channel": ("per-channel, abs-max", False, None),                    # standard
         "tensor": ("per-tensor, abs-max", True, None),                       # one scale per layer, common on small hardware
         "clip": ("per-channel, 99th-percentile clipping", False, 0.99)}      # outlier-clipping calibration used for time-series PTQ

# ---- unlearning: time-series methods, 400 steps, one fixed standard setting per method (no search, no tuning) ----
STEPS, BATCH_U = 400, 64
METHODS = {"RandLabel": 1e-3, "ZeroLabel": 1e-3, "NoisyLabel": 1e-3, "MaxLoss": 1e-4, "TSU": 1e-3, "GaussAmnesiac": 0.1, "Blindspot": 1e-3}
TECHS = {a: ["channel", "tensor", "clip"] for a in ["dlinear", "mlp", "patchtst"]}

CTRL_EVERY, CTRL_N, RETAIN_GUARD = 2, 256, 1.10
CTRL_Q = 0.75                                       # target = 75th percentile (round 2 used the median) of how well the start model does on never-trained clients
# controller: look every 2 steps at 256 forget and 256 retain windows; stop if retain loss exceeds 1.10 x its start value
METHODS_R3 = {"TPU": 1e-3, "MaxLoss": 1e-4}          # our method and the best existing one from round 2, one fixed setting each
TPU_CFG = dict(opt="nsgd", lr=0.003, lam=1.0, frac=0.10, steps=1600, grow_every=200)   # TPU settings: optimiser, step, retain weight, share of weights per layer that may change
TECHS_BELOW4 = ["channel"]                          # INT3 is run with the standard per-channel quantizer only

# ---- rules fixed BEFORE running (docs/DESIGN.md) ----
STUCK_BELOW = 0.01                       # fewer than 1% of the integer weights changed -> the method did nothing
TRACK_ABS, TRACK_REL, TRACK_RETAIN = 0.25, 0.15, 0.10   # "tracks full precision": forget score within max(0.25, 15% of the full-precision score) and retain error at most 0.10 worse than the full-precision run of the same method and controller setting
FORGOTTEN_RANGE, MAX_RETAIN_DAMAGE = (0.8, 1.2), 1.10     # "forgot correctly"
