import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------- tiny utilities ----------
def lengths_to_mask(lengths, max_len=None, device=None):
    # True = pad (what MultiheadAttention expects)
    B = lengths.size(0)
    if max_len is None:
        max_len = int(lengths.max())
    ar = torch.arange(max_len, device=device or lengths.device).unsqueeze(0).expand(B, -1)
    return ar >= lengths.unsqueeze(1)  # [B, max_len] bool

# ---------- building blocks ----------
class MLP(nn.Module):
    def __init__(self, d, hidden_mult=2, dropout=0.0):
        super().__init__()
        h = int(d * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, d)
        )
    def forward(self, x):
        return self.net(x)

class MAB(nn.Module):
    """Multihead Attention Block: LN(x + Attn(Q,K,V)) -> LN(.) + MLP"""
    def __init__(self, d_model, nhead=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, hidden_mult=2, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, key_padding_mask=None):
        # key_padding_mask: [B, N_kv] where True marks padding positions in K/V
        y, _ = self.attn(Q, K, V, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(Q + self.drop(y))
        y = self.mlp(x)
        return self.ln2(x + self.drop(y))

class SAB(nn.Module):
    """Self-Attention Block: MAB(X, X, X)"""
    def __init__(self, d_model, nhead=4, dropout=0.0):
        super().__init__()
        self.mab = MAB(d_model, nhead, dropout)
    def forward(self, X, key_padding_mask=None):
        return self.mab(X, X, X, key_padding_mask=key_padding_mask)

class PMA(nn.Module):
    """Pooling by Multihead Attention (k learnable seeds)"""
    def __init__(self, d_model, nhead=4, k=1, dropout=0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d_model) * 0.02)  # seeds
        self.mab = MAB(d_model, nhead, dropout)
    def forward(self, X, key_padding_mask=None):
        B = X.size(0)
        S = self.S.expand(B, -1, -1)  # [B, k, D]
        # queries = seeds, keys/values = set elements
        return self.mab(S, X, X, key_padding_mask=key_padding_mask)  # [B, k, D]



# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# plt.ion()  # turn on interactive mode
#
# def plot_pointcloud(pc, colors=None, sample=5000):
#     """
#     pc: (N,3) numpy array of XYZ points
#     colors: (N,3) uint8 or float array of RGB values (optional)
#     sample: max number of points to plot for speed
#     """
#     if pc.size == 0:
#         print("Empty point cloud")
#         return
#
#     # Subsample if too many points
#     if pc.shape[0] > sample:
#         idx = np.random.choice(pc.shape[0], sample, replace=False)
#         pc = pc[idx]
#         if colors is not None:
#             colors = colors[idx]
#
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection="3d")
#     if colors is not None:
#         ax.scatter(
#             pc[:, 0],
#             pc[:, 1],
#             pc[:, 2],
#             c=colors / 255.0 if colors.dtype != float else colors,
#             s=1,
#         )
#     else:
#         ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c="blue")
#
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.view_init(elev=30, azim=45)
#     plt.tight_layout()
#     plt.show()



class PointNet2SetEncoder(nn.Module):

    def __init__(self, in_feat_dim=0, z_dim=64, n_frames=3):

        super().__init__()
        d_model = 64
        self.in_proj = nn.Linear(6, d_model)
        self.sabs = nn.ModuleList([SAB(d_model, 4, 0.0) for _ in range(2)])
        self.pma = PMA(d_model, 4, k=1, dropout=0.0)

        self.temp_aggr = nn.Linear(d_model * n_frames, z_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(z_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, z_dim)
        )

    def forward(self, pc):

        xyz = pc['pc']
        rgb = pc['pc_rgb']
        B, T, N, _ = xyz.shape

        x = torch.cat([xyz.view(B*T, N, -1), rgb.view(B*T, N, -1)], -1)

        lengths = (rgb.sum(-1) > 0).sum(-1).view(B * T)
        pad_mask = lengths_to_mask(lengths, max_len=N, device=x.device)

        H = self.in_proj(x)                       # [B, N, D]
        for sab in self.sabs:
            H = sab(H, key_padding_mask=pad_mask)     # [B, N, D]

        P = self.pma(H, key_padding_mask=pad_mask)    # [B, k, D]
        P = P.squeeze(1).contiguous()                              # [B*T, 32]
        P = P.view(B, T, -1).reshape(B, T * P.size(-1))

        h = self.temp_aggr(P) #.view(B, -1))
        y = self.mlp2(h)
        return y
















