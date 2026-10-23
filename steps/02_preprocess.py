"""Keep 20 clients, split by time, standardise per client (train period only), cut sliding windows."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, json
import config as C
from src.data import load_raw, split_points, standardise, build_sets
X = load_raw()[:, :20].T.copy(); np.save(C.INTERIM / "clients20_raw.npy", X)
t_tr, t_va = split_points(X.shape[1]); Xs, _, _ = standardise(X, t_tr); D = build_sets(Xs, t_tr, t_va); torch.save(D, C.PROCESSED / "windows.pt")
info = dict(hours=X.shape[1], train_end=t_tr, val_end=t_va, lookback=C.L, horizon=C.H, **{k: len(v) for k, v in D.items() if k.startswith("X")})
json.dump(info, open(C.RES / "data_summary.json", "w"), indent=1); print(info)
