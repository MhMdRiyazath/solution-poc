import copy, torch, torch.nn.functional as F
import config as C
from .models import qlinears, make_model
from .quantize import project
from .metrics import mse, ctrl_sets, ctrl_target

def saliency_masks(m, params, D, frac, n_batches=8):
    """Which integer weights to allow to change: per layer, the `frac` of weights whose squared gradient on forget windows is largest relative to their squared gradient on retain windows
    (weights that matter for the forgotten client but little for the kept clients). Computed once on the model being unlearned; no extra model copy is kept."""
    m.eval(); gf = [torch.zeros_like(p) for p in params]; gr = [torch.zeros_like(p) for p in params]
    for _ in range(n_batches):
        bi = torch.randint(0, len(D["Xr"]), (C.BATCH_U,)); bf = torch.randint(0, len(D["Xf"]), (C.BATCH_U,))
        for acc, (x, y) in ((gf, (D["Xf"][bf], D["Yf"][bf])), (gr, (D["Xr"][bi], D["Yr"][bi]))):
            for g, gd in zip(acc, torch.autograd.grad(F.mse_loss(m(x), y), params)): g += gd ** 2
    masks = []
    for a, b in zip(gf, gr):
        s = (a / (b + b.mean() * 1e-3 + 1e-30)).flatten(); k = max(1, int(frac * s.numel())); thr = torch.topk(s, k).values[-1]; masks.append((a / (b + b.mean() * 1e-3 + 1e-30) >= thr).float())
    return masks

def unlearn(m0, method, hp, D, arch, quantized, seed, rounding="nearest", ctrl=False):
    """Time-series unlearning methods, client 10 = forget set, the other 9 clients = retain set. hp = learning rate (noise level for GaussAmnesiac). Adam, 400 steps, batch 64.
      RandLabel      forget windows trained toward random targets, plus the retain loss                      (TS-Unlearn family of label-based baselines)
      ZeroLabel      forget windows trained toward zeros, plus the retain loss                               (same family)
      NoisyLabel     forget windows trained toward noisy copies of their targets, plus the retain loss      (TS-Unlearn baseline)
      MaxLoss        gradient ascent on the forget windows (capped), plus the retain loss
      TSU            TS-Unlearn dual objective: 0.5 x retain loss + 0.5 x loss toward noisy forget targets   (fixed Pareto weight 0.5)
      GaussAmnesiac  Gaussian noise added to the weights (relative std = hp), then 200 retain-only repair steps
      TPU            (ours) Target-seeking, Protected-subset Unlearning: only the top 10% of weights per layer by forget/retain gradient ratio may change; the loss is a one-sided pull of the forget loss up to the
                     reference-free target plus the retain loss relative to its start value. Always run with the controller.
      Blindspot      distillation: stay close to the original model on retain windows, copy an untrained network on forget windows
    quantized=True : the model is the deployed quantized one and no full-precision copy exists. Only its integer Linear weights may change; after every step they are put back on the integer grid
                     (scales fixed, biases / norms frozen) with `rounding` = 'nearest' (the usual way) or 'stochastic' (the proposed change).
    quantized=False: full-precision model (reference only), all parameters train.
    ctrl=True      : controller. Every 2 steps look at 256 forget and 256 retain windows; stop when the forget loss reaches the reference-free target (ctrl_target) or the retain loss exceeds 1.10 x its start value.
                     Needs no retrained model and no full-precision copy. Not used for GaussAmnesiac (its repair length is fixed)."""
    m = copy.deepcopy(m0); torch.manual_seed(seed + 7); lr = 3e-4 if method == "GaussAmnesiac" else hp
    params = [l.weight for l in qlinears(m)] if quantized else list(m.parameters())
    if method == "TPU": params = [l.weight for l in qlinears(m)]
    for p in m.parameters(): p.requires_grad_(any(p is q for q in params))
    teacher = None
    if method == "Blindspot": torch.manual_seed(1000 + seed); teacher = make_model(arch).eval(); torch.manual_seed(seed + 7)
    steps = C.STEPS; proj = lambda: project(m, rounding) if quantized else None
    if method == "GaussAmnesiac":
        with torch.no_grad():
            for p in params: p.add_(torch.randn_like(p) * hp * p.std())
        proj(); steps = 200
    use_stop = ctrl and method != "GaussAmnesiac"
    if method == "TPU": steps = C.TPU_CFG.get("steps", steps)
    if use_stop:
        xs, ys, xr_, yr_ = ctrl_sets(D); target = ctrl_target(m0, D, xs, ys); rguard = C.RETAIN_GUARD * mse(m0, xr_, yr_)
    if method == "TPU":
        assert use_stop, "TPU needs the controller (its target)"; r0 = mse(m0, xr_, yr_); masks = saliency_masks(m, params, D, C.TPU_CFG["frac"]); m.train()
    opt = torch.optim.Adam(params, lr); q0 = [l.weight.detach().clone() for l in qlinears(m0)]; orig = copy.deepcopy(m0).eval(); done = steps
    for st in range(steps):
        if method == "TPU" and st > 0 and st % C.TPU_CFG["grow_every"] == 0 and C.TPU_CFG["frac"] * 2 ** (st // C.TPU_CFG["grow_every"]) <= 2 * C.TPU_CFG["frac"] * 2 ** (st // C.TPU_CFG["grow_every"] - 1):   # still short of the target: let twice as many weights change
            masks = saliency_masks(m, params, D, min(1.0, C.TPU_CFG["frac"] * 2 ** (st // C.TPU_CFG["grow_every"])))
        m.train(); bi = torch.randint(0, len(D["Xr"]), (C.BATCH_U,)); bf = torch.randint(0, len(D["Xf"]), (C.BATCH_U,))
        xr, yr, xf, yf = D["Xr"][bi], D["Yr"][bi], D["Xf"][bf], D["Yf"][bf]; lret = F.mse_loss(m(xr), yr)
        noisy = yf + torch.randn_like(yf) * yf.std(1, keepdim=True)
        if method == "GaussAmnesiac": loss = lret
        elif method == "RandLabel": loss = lret + F.mse_loss(m(xf), torch.randn_like(yf))
        elif method == "ZeroLabel": loss = lret + F.mse_loss(m(xf), torch.zeros_like(yf))
        elif method == "NoisyLabel": loss = lret + F.mse_loss(m(xf), noisy)
        elif method == "TSU": loss = 0.5 * lret + 0.5 * F.mse_loss(m(xf), noisy)
        elif method == "MaxLoss": loss = lret - 0.5 * torch.clamp(F.mse_loss(m(xf), yf), max=5.0)
        elif method == "TPU": loss = torch.relu(1 - F.mse_loss(m(xf), yf) / target) ** 2 + C.TPU_CFG["lam"] * lret / r0
        elif method == "Blindspot":
            with torch.no_grad(): to_r, to_f = orig(xr), teacher(xf)
            loss = F.mse_loss(m(xr), to_r) + lret + F.mse_loss(m(xf), to_f)
        else: raise ValueError(method)
        opt.zero_grad(); loss.backward()
        if method == "TPU":
            for p, mk in zip(params, masks): p.grad.mul_(mk)
        if method == "TPU" and C.TPU_CFG["opt"] == "nsgd":      # normalised gradient step: total step length = eta x total weight norm, spread in proportion to the gradient
            with torch.no_grad():
                gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params)) + 1e-12; wn = torch.sqrt(sum((p ** 2).sum() for p in params))
                for p in params: p.add_(p.grad, alpha=-(C.TPU_CFG["lr"] * wn / gn).item())
            proj()
        else: torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); proj()
        if use_stop and (st + 1) % C.CTRL_EVERY == 0 and (mse(m, xs, ys) >= target or mse(m, xr_, yr_) > rguard): done = st + 1; break
    changed = sum((a != l.weight.detach()).sum().item() for a, l in zip(q0, qlinears(m))) / sum(a.numel() for a in q0)
    return m.eval(), changed, done
