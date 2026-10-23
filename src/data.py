import gzip, numpy as np, torch
import config as C

def load_raw():
    """Electricity file: rows = hours, columns = clients (321). Returns [hours, clients]."""
    return np.loadtxt(gzip.open(C.RAW_FILE), delimiter=",", dtype=np.float32)

def split_points(T): return int(T * C.TRAIN_END), int(T * C.VAL_END)

def standardise(X, t_tr):
    """Per-client z-score using the TRAIN period only (no leakage from validation / test)."""
    mu, sd = X[:, :t_tr].mean(1, keepdims=True), X[:, :t_tr].std(1, keepdims=True) + 1e-6
    return (X - mu) / sd, mu, sd

def windows(X, c, lo, hi, stride):
    s = X[c, lo:hi]; idx = np.arange(0, hi - lo - C.L - C.H + 1, stride)
    return torch.tensor(np.stack([s[i:i + C.L] for i in idx])), torch.tensor(np.stack([s[i + C.L:i + C.L + C.H] for i in idx]))

def cat_windows(X, clients, lo, hi, stride):
    xs, ys = zip(*[windows(X, c, lo, hi, stride) for c in clients]); return torch.cat(xs), torch.cat(ys)

def build_sets(X, t_tr, t_va):
    """Window sets. Names: r = retain clients (9), f = forget client (10), c = control clients never trained on (11-20)."""
    T = X.shape[1]; retain = [c for c in C.TRAIN_CLIENTS if c != C.FORGET]; D = {}
    D["Xr"], D["Yr"] = cat_windows(X, retain, 0, t_tr, C.WINDOW_STRIDE)
    D["Xf"], D["Yf"] = windows(X, C.FORGET, 0, t_tr, C.WINDOW_STRIDE)
    fi = torch.randperm(len(D["Xf"]), generator=torch.Generator().manual_seed(0))[:1500]; D["Xf_e"], D["Yf_e"] = D["Xf"][fi], D["Yf"][fi]   # forget windows used for evaluation
    D["Xc"], D["Yc"] = cat_windows(X, C.CONTROL, 0, t_tr, 60)
    D["Xrt"], D["Yrt"] = cat_windows(X, retain, t_va, T, 24)          # retain test
    D["Xrv"], D["Yrv"] = cat_windows(X, retain, t_tr, t_va, 24)       # retain validation (used only by the controller guard)
    D["Xft"], D["Yft"] = windows(X, C.FORGET, t_va, T, 24)            # forget-client test
    D["Xct"], D["Yct"] = cat_windows(X, C.CONTROL, t_va, T, 24)       # control-client test (used only by the reference-free stop target)
    return D

def load_processed(): return torch.load(C.PROCESSED / "windows.pt")
