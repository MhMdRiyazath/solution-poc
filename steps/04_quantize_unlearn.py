"""Quantize -> unlearn the quantized model (no full-precision copy) -> judge.
   Methods: TPU (ours) and MaxLoss (best existing method of the earlier rounds), both with the controller, one fixed setting each.
   Variants: fp (full-precision reference), nearest (usual rounding), stochastic (proposed rounding). Quantizers: per-channel, per-tensor, 99th-percentile clipping; INT8, INT6, INT4.
   The retrained model (same precision) is only the judge."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, csv, time, torch, config as C
from src.data import load_processed
from src.models import make_model
from src.quantize import quantize
from src.metrics import reference, evaluate, verify, ctrl_sets, ctrl_target
from src.unlearn import unlearn
torch.set_num_threads(1)
ap = argparse.ArgumentParser(); ap.add_argument("--arch", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--bits", default="8,6,4"); a = ap.parse_args()
bits = [int(b) for b in a.bits.split(",")]; D = load_processed(); t0 = time.time(); xs, ys, xr, yr = ctrl_sets(D)
out = open(C.RES / f"raw_{a.arch}_seed{a.seed}.csv", "w", newline="")
w = csv.DictWriter(out, fieldnames=["arch", "seed", "tech", "bits", "rounding", "method", "hp", "changed", "steps", "seconds", "forget_score", "mia_auc", "retain_ratio", "passed", "gap", "s_hat", "retain_est", "target", "start_f"]); w.writeheader()
def load(name): m = make_model(a.arch); m.load_state_dict(torch.load(C.CKPT / a.arch / f"seed{a.seed}_{name}.pt")); return m.eval()
orig, retrained = load("original"), load("retrained")
conds = [("fp", 32, orig, retrained, False)] + [(t, b, quantize(orig, t, b), quantize(retrained, t, b), True) for t in C.TECHS[a.arch] for b in bits]
for tech, b, m0, g, quantized in conds:
    ref = reference(m0, g, D); tgt = ctrl_target(m0, D, xs, ys)
    for rounding in (["stochastic", "nearest"] if quantized else ["none"]):
        for method, hp in C.METHODS_R3.items():
            t1 = time.time(); m, ch, steps = unlearn(m0, method, hp, D, a.arch, quantized, a.seed, rounding if quantized else "nearest", True); sec = time.time() - t1
            e = evaluate(m, ref, D); v = verify(m0, m, D, xs, ys, xr, yr)
            w.writerow(dict(arch=a.arch, seed=a.seed, tech=tech, bits=b, rounding=rounding, method=method, hp=hp, changed=ch if quantized else "", steps=steps, seconds=round(sec, 2), gap=ref["gap"], target=tgt, start_f=ref["start_f"], **e, **v)); out.flush()
    print(f"[{time.time()-t0:.0f}s] {a.arch} seed {a.seed} {tech} {b}-bit done", flush=True)
