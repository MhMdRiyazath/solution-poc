import numpy as np, torch
from sklearn.metrics import roc_auc_score
import config as C

@torch.no_grad()
def win_loss(m, x, y):
    m.eval(); return torch.cat([((m(x[i:i + 1024]) - y[i:i + 1024]) ** 2).mean(1) for i in range(0, len(x), 1024)])
def mse(m, x, y): return win_loss(m, x, y).mean().item()

def reference(start, retrained, D):
    """start = model before unlearning, retrained = model trained without client 10; both in the SAME precision as the model under test."""
    fg, cg = win_loss(retrained, D["Xf_e"], D["Yf_e"]), win_loss(retrained, D["Xc"], D["Yc"]); sf = mse(start, D["Xf_e"], D["Yf_e"])
    return dict(start_f=sf, gold_f=fg.mean().item(), start_retain=mse(start, D["Xrt"], D["Yrt"]), gap=(fg.mean().item() - sf) / fg.mean().item(), fg=fg, cg=cg)

def evaluate(m, ref, D):
    """forget_score: 0 = still remembers client 10 (like the start model), 1 = as ignorant as the retrained model.
    mia_auc: attacker scores windows by (retrained loss - model loss); client-10 windows vs never-seen clients; 0.5 = cannot tell them apart.
    retain_ratio: retain-client test error divided by the start model's."""
    f, c = win_loss(m, D["Xf_e"], D["Yf_e"]), win_loss(m, D["Xc"], D["Yc"])
    score = (f.mean().item() - ref["start_f"]) / (ref["gold_f"] - ref["start_f"])
    s = torch.cat([ref["fg"] - f, ref["cg"] - c]).numpy(); lab = np.r_[np.ones(len(f)), np.zeros(len(c))]
    auc = roc_auc_score(lab, s) if np.ptp(s) > 0 else 0.5; ratio = mse(m, D["Xrt"], D["Yrt"]) / ref["start_retain"]
    return dict(forget_score=score, mia_auc=auc, retain_ratio=ratio, passed=bool(C.FORGOTTEN_RANGE[0] <= score <= C.FORGOTTEN_RANGE[1] and ratio <= C.MAX_RETAIN_DAMAGE))


# ---- reference-free pieces (no retrained model, no full-precision copy) ----
def naive_mse(x, y):
    """Previous-day forecast (repeat the last 24 input hours): a model-free yardstick for how hard a set of windows is."""
    return ((x[:, -C.H:] - y) ** 2).mean().item()

def skill_median(m0, D):
    """Quantile CTRL_Q (0.75) over the never-trained clients (11-20) of  model loss / previous-day-forecast loss  = how well a model does on a client it has never seen."""
    n = len(D["Xc"]) // len(C.CONTROL); r = []
    for i in range(len(C.CONTROL)):
        x, y = D["Xc"][i * n:(i + 1) * n], D["Yc"][i * n:(i + 1) * n]; r.append(mse(m0, x, y) / naive_mse(x, y))
    return float(np.quantile(r, C.CTRL_Q))

def ctrl_sets(D):
    """Fixed small subsets used by the controller: forget windows (training period) and retain windows from the validation period (held out from training, like the final test)."""
    g = torch.Generator().manual_seed(123); fi = torch.randperm(len(D["Xf"]), generator=g)[:C.CTRL_N]; ri = torch.randperm(len(D["Xrv"]), generator=g)[:C.CTRL_N]
    return D["Xf"][fi], D["Yf"][fi], D["Xrv"][ri], D["Yrv"][ri]

def ctrl_target(m0, D, xs, ys):
    """Forget-loss target without a retrained model: what the start model would score on the forget windows if they came from a typical unseen client."""
    return skill_median(m0, D) * naive_mse(xs, ys)

def verify(m0, m, D, xs, ys, xr, yr):
    """Reference-free check of forgetting: s_hat = how far the forget loss moved from start toward the target (1 = reached it); retain_est = retain loss / start retain loss. Same data as the controller, no retrained model."""
    t = ctrl_target(m0, D, xs, ys); f0, f1 = mse(m0, xs, ys), mse(m, xs, ys)
    return dict(s_hat=(f1 - f0) / (t - f0) if t != f0 else float("nan"), retain_est=mse(m, xr, yr) / mse(m0, xr, yr))
