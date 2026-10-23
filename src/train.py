import torch, torch.nn.functional as F
import config as C
from .models import make_model

def train(arch, seed, x, y, epochs=C.EPOCHS):
    """Plain full-precision training, fixed number of epochs, cosine learning-rate schedule."""
    torch.manual_seed(seed); m = make_model(arch); n = len(x); opt = torch.optim.Adam(m.parameters(), C.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * ((n + C.BATCH - 1) // C.BATCH))
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, C.BATCH):
            b = perm[i:i + C.BATCH]; loss = F.mse_loss(m(x[b]), y[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()
