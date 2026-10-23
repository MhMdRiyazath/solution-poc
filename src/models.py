import torch, torch.nn as nn, torch.nn.functional as F
import config as C

class QLinear(nn.Linear):
    """nn.Linear that remembers its quantization scale and bit range (set by quantize.py)."""
    def __init__(s, *a, **k): super().__init__(*a, **k); s.qmax = None; s.register_buffer("scale", torch.ones(1))

class Block(nn.Module):
    def __init__(s, d, h, ff, p):
        super().__init__(); s.h = h; s.q, s.k, s.v, s.o = [QLinear(d, d) for _ in range(4)]
        s.f1, s.f2 = QLinear(d, ff), QLinear(ff, d); s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d); s.dr = nn.Dropout(p)
    def forward(s, x):
        B, N, D = x.shape; y = s.n1(x)
        q, k, v = [m(y).view(B, N, s.h, D // s.h).transpose(1, 2) for m in (s.q, s.k, s.v)]
        att = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + s.dr(s.o(att)); return x + s.dr(s.f2(F.gelu(s.f1(s.n2(x)))))

class PatchTST(nn.Module):
    """Small PatchTST (Nie et al. 2023): instance-normalise -> patch -> transformer encoder -> linear head. Channel independent."""
    def __init__(s, d=64, h=4, layers=2, ff=128, p=0.1):
        super().__init__(); n = (C.L - C.PATCH) // C.STRIDE + 1
        s.emb = QLinear(C.PATCH, d); s.pos = nn.Parameter(torch.randn(1, n, d) * .02)
        s.blocks = nn.ModuleList([Block(d, h, ff, p) for _ in range(layers)]); s.head = QLinear(n * d, C.H)
    def forward(s, x):
        m, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-5; z = (x - m) / sd
        h = s.emb(z.unfold(1, C.PATCH, C.STRIDE)) + s.pos
        for b in s.blocks: h = b(h)
        return s.head(h.flatten(1)) * sd + m

class DLinear(nn.Module):
    """DLinear (Zeng et al. 2023): moving-average trend + remainder, one Linear layer each."""
    def __init__(s, k=25): super().__init__(); s.k = k; s.seas = QLinear(C.L, C.H); s.trend = QLinear(C.L, C.H)
    def forward(s, x):
        pad = (s.k - 1) // 2; trend = F.avg_pool1d(F.pad(x.unsqueeze(1), (pad, pad), mode="replicate"), s.k, 1).squeeze(1)
        return s.seas(x - trend) + s.trend(trend)

class MLP(nn.Module):
    """Plain MLP forecaster with instance normalisation (N-HiTS-style block, no pooling)."""
    def __init__(s, d=512):
        super().__init__(); s.l1, s.l2, s.l3 = QLinear(C.L, d), QLinear(d, d), QLinear(d, C.H)
    def forward(s, x):
        m, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-5; z = (x - m) / sd
        return s.l3(F.gelu(s.l2(F.gelu(s.l1(z))))) * sd + m

MODELS = {"patchtst": PatchTST, "dlinear": DLinear, "mlp": MLP}
def make_model(arch): return MODELS[arch]()
def qlinears(m): return [x for x in m.modules() if isinstance(x, QLinear)]
