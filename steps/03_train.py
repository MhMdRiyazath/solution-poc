"""Train two models per architecture and seed:  original = all 10 clients;  retrained = the 9 other clients only (client 10 never seen = the ideal 'forgotten' model)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, torch, config as C
from src.data import load_processed
from src.train import train
ap = argparse.ArgumentParser(); ap.add_argument("--arch", required=True); a = ap.parse_args(); D = load_processed()
for seed in C.SEEDS[a.arch]:
    for name, (x, y) in {"original": (torch.cat([D["Xr"], D["Xf"]]), torch.cat([D["Yr"], D["Yf"]])), "retrained": (D["Xr"], D["Yr"])}.items():
        f = C.CKPT / a.arch / f"seed{seed}_{name}.pt"; f.parent.mkdir(exist_ok=True)
        if not f.exists(): torch.save(train(a.arch, seed, x, y).state_dict(), f); print("trained", f.name, flush=True)
