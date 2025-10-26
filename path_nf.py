# ===========================
# Full script: NF-pushed SDE on Lorenz data -> plot.jpg
# ===========================
# - Data: sample trajectories from a stochastic Lorenz SDE, z-normalize, add tiny noise
# - Model: latent diagonal OU SDE; x_t = Flow_t(y_t) (time-conditioned RealNVP)
# - Training: path-wise MLE = OU path log-prob in y + sum_t log|det ∂f^{-1}(x_t,t)|
# ===========================

from typing import Any, Sequence, Callable
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from torch import distributions as D

import matplotlib
matplotlib.use("Agg")  # headless-safe for saving to file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D projection is registered)
from tqdm import trange


torch.set_float32_matmul_precision("high") if hasattr(torch, "set_float32_matmul_precision") else None
torch.set_default_dtype(torch.float32)

def solve_sde(
        sde: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int
) -> Tensor:
    tt = torch.linspace(ts, tf, n_steps + 1, device=z.device, dtype=z.dtype)[:-1]
    dt = (tf - ts) / n_steps
    dt_2 = (abs(dt)) ** 0.5
    path = [z]
    for t in tt:
        f, g = sde(z, t)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2
        path.append(z)
    return torch.stack(path)  # (n_steps+1, B, D)

# -------------------------
# Base class & Stochastic Lorenz SDE
# -------------------------
class SDE(nn.Module, ABC):
    @abstractmethod
    def drift(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        raise NotImplementedError
    @abstractmethod
    def vol(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        raise NotImplementedError
    def forward(self, z: Tensor, t: Tensor, *args: Any) -> tuple[Tensor, Tensor]:
        return self.drift(z, t, *args), self.vol(z, t, *args)

class StochasticLorenzSDE(SDE):
    def __init__(self, a: Sequence = (10., 28., 8. / 3.), b: Sequence = (0.15, 0.15, 0.15)):
        super().__init__()
        self.a = a
        self.b = b
    def drift(self, x: Tensor, t: Tensor, *args) -> Tensor:
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        a1, a2, a3 = self.a
        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)
    def vol(self, x: Tensor, t: Tensor, *args) -> Tensor:
        x1, x2, x3 = torch.split(x, [1, 1, 1], dim=1)
        b1, b2, b3 = self.b
        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3
        return torch.cat([g1, g2, g3], dim=1)

# -------------------------
# Data generator from Lorenz SDE (normalized + small noise)
# -------------------------
@torch.no_grad()
def gen_data(
        batch_size: int,
        ts: float,
        tf: float,
        n_steps: int,
        noise_std: float,
        n_inner_steps: int = 100,
        device: str = "cuda"
) -> tuple[Tensor, Tensor]:
    sde = StochasticLorenzSDE().to(device)
    z0 = torch.randn(batch_size, 3, device=device)
    zs = solve_sde(sde, z0, ts, tf, n_steps=n_steps * n_inner_steps)  # (n_inner*n_steps+1, B, 3)
    zs = zs[::n_inner_steps]                  # subsample to n_steps+1
    zs = zs.permute(1, 0, 2).contiguous()     # (B, T, 3)

    mean = torch.mean(zs, dim=(0, 1), keepdim=True)
    std = torch.std(zs, dim=(0, 1), keepdim=True).clamp_min(1e-6)

    eps = torch.randn_like(zs)
    xs = (zs - mean) / std + noise_std * eps  # normalized + tiny noise

    tt = torch.linspace(ts, tf, n_steps + 1, device=device, dtype=zs.dtype)
    tt = tt[None, :, None].repeat(batch_size, 1, 1)  # (B, T, 1)
    return xs, tt

# -------------------------
# NF-pushed SDE: OU in latent y, time-conditioned RealNVP to x
# -------------------------
def diag_gauss_logprob(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    var = var.clamp_min(1e-10)
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, device=x.device, dtype=x.dtype))
    return -0.5 * (torch.log(var) + log2pi + (x - mean) ** 2 / var).sum(dim=-1)

class TimeEmbed(nn.Module):
    def __init__(self, dim: int = 64, max_freq: float = 16.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(0., torch.log(torch.tensor(max_freq, dtype=torch.float32)), dim // 2))
        self.register_buffer("freqs", freqs)  # (dim//2,)
    def forward(self, t: Tensor) -> Tensor:  # t: (B,1) in [0,1]
        t = t.to(self.freqs.dtype)
        ang = t * self.freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class AffineCoupling(nn.Module):
    def __init__(self, D: int, hidden: int, tdim: int, mask: Tensor, clamp: float = 1.5):
        super().__init__()
        self.register_buffer("mask", mask.view(1, -1).float())  # 0/1 selector
        self.temb = TimeEmbed(tdim)
        self.net = MLP(in_dim=D + tdim, hidden=hidden, out_dim=2 * D)
        self.clamp = clamp
    def forward(self, x: Tensor, t: Tensor, inverse: bool = False):
        m = self.mask
        xa = x * m
        h = torch.cat([xa, self.temb(t)], dim=-1)
        s, b = self.net(h).chunk(2, dim=-1)
        s = torch.tanh(s) * self.clamp
        # only act on complement
        comp = (1. - m)
        s = s * comp
        b = b * comp
        if not inverse:
            y = xa + comp * (x * torch.exp(s) + b)
            logdet = s.sum(dim=-1)
        else:
            y = xa + comp * ((x - b) * torch.exp(-s))
            logdet = (-s).sum(dim=-1)
        return y, logdet

class TimeCondRealNVP(nn.Module):
    def __init__(self, D: int, hidden: int = 160, n_layers: int = 6, tdim: int = 64):
        super().__init__()
        base_mask = torch.tensor([1 if i % 2 == 0 else 0 for i in range(D)])
        masks = [(base_mask if k % 2 == 0 else 1 - base_mask) for k in range(n_layers)]
        self.layers = nn.ModuleList([AffineCoupling(D, hidden, tdim, m) for m in masks])
    # y -> x
    def forward(self, y: Tensor, t: Tensor) -> Tensor:
        x = y
        for layer in self.layers:
            x, _ = layer(x, t, inverse=False)
        return x
    # x -> y and sum log|det J|
    def inverse(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        y = x
        logdet = x.new_zeros(x.size(0))
        for layer in reversed(self.layers):
            y, ld = layer(y, t, inverse=True)
            logdet = logdet + ld
        return y, logdet

class DiagOUSDE(nn.Module):
    """
    dY = κ ⊙ (μ - Y) dt + σ ⊙ dW_t  (diagonal OU; exact discrete transitions)
    """
    def __init__(self, D: int, init_mu: float = 0.0, init_logk: float = -0.7, init_logs: float = -0.7):
        super().__init__()
        self.mu = nn.Parameter(torch.full((D,), init_mu))
        self.log_kappa = nn.Parameter(torch.full((D,), init_logk))
        self.log_sigma = nn.Parameter(torch.full((D,), init_logs))
    def _params(self):
        kappa = torch.nn.functional.softplus(self.log_kappa) + 1e-6  # >0
        sigma = torch.nn.functional.softplus(self.log_sigma) + 1e-6  # >0
        mu = self.mu
        return mu, kappa, sigma
    @torch.no_grad()
    def sample_path(self, ts_grid: Tensor, n: int) -> Tensor:
        device = ts_grid.device
        D = self.mu.numel()
        mu, kappa, sigma = self._params()
        mu = mu.to(device); kappa = kappa.to(device); sigma = sigma.to(device)
        T = ts_grid.size(0)
        y = torch.zeros(n, T, D, device=device)
        # stationary prior for Y_0
        var0 = (sigma**2) / (2.0 * kappa)
        y[:, 0, :] = mu + torch.randn(n, D, device=device) * var0.sqrt()
        for k in range(T - 1):
            dt = (ts_grid[k+1] - ts_grid[k]).clamp(min=1e-6)  # (1,)
            Ad = torch.exp(-kappa * dt)                      # (D,)
            mean = mu + Ad * (y[:, k, :] - mu)               # (n, D)
            q = (sigma**2) * (1.0 - torch.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
            y[:, k+1, :] = mean + torch.randn_like(mean) * q.sqrt()
        return y
    def path_log_prob(self, y: Tensor, ts_batch: Tensor) -> Tensor:
        B, T, D = y.shape
        if ts_batch.dim() == 2:
            ts_batch = ts_batch[None, :, :].expand(B, -1, -1)
        mu, kappa, sigma = self._params()
        mu = mu[None, None, :].to(y.device)
        kappa = kappa[None, None, :].to(y.device)
        sigma = sigma[None, None, :].to(y.device)
        # initial density (stationary)
        var0 = (sigma**2) / (2.0 * kappa)
        lp0 = diag_gauss_logprob(y[:, 0, :], mu[:, 0, :], var0[:, 0, :])  # (B,)
        # transitions
        t0, t1 = ts_batch[:, :-1, :], ts_batch[:, 1:, :]
        dt = (t1 - t0).clamp(min=1e-6)                       # (B,T-1,1)
        Ad = torch.exp(-kappa * dt)                          # (B,T-1,D)
        mean = mu + Ad * (y[:, :-1, :] - mu)                 # (B,T-1,D)
        q = (sigma**2) * (1.0 - torch.exp(-2.0 * kappa * dt)) / (2.0 * kappa)  # (B,T-1,D)
        lp_trans = diag_gauss_logprob(y[:, 1:, :], mean, q).sum(dim=1)
        return lp0 + lp_trans

class NF_SDE_Model(nn.Module):
    def __init__(self, D: int, hidden: int = 160, n_layers: int = 6, tdim: int = 64):
        super().__init__()
        self.flow = TimeCondRealNVP(D, hidden=hidden, n_layers=n_layers, tdim=tdim)
        self.ou = DiagOUSDE(D)
    def log_prob_paths(self, x: Tensor, ts_batch: Tensor) -> Tensor:
        B, T, D = x.shape
        if ts_batch.dim() == 2:
            ts_batch = ts_batch[None, :, :].expand(B, -1, -1)
        xf = x.reshape(B * T, D)
        tf = ts_batch.reshape(B * T, 1)
        yf, logdetf = self.flow.inverse(xf, tf)              # (B*T,D), (B*T,)
        y = yf.reshape(B, T, D)
        logdet_seq = logdetf.reshape(B, T).sum(dim=1)        # (B,)
        lp_y = self.ou.path_log_prob(y, ts_batch)            # (B,)
        return lp_y + logdet_seq

    
    @torch.no_grad()
    def sample_paths(self, ts_grid: Tensor, n_paths: int = 6) -> Tensor:
        y = self.ou.sample_path(ts_grid, n_paths)            # (n, T, D)
        T = ts_grid.size(0)
        D = y.size(-1)
        yf = y.reshape(-1, D)
        tf = ts_grid[None, :, :].expand(n_paths, -1, -1).reshape(-1, 1)
        x = self.flow.forward(yf, tf).reshape(n_paths, T, D)
        return x

def train_nf_sde(model: NF_SDE_Model, xs: Tensor, ts: Tensor,
                 iters: int = 2000, lr: float = 2e-3, batch_seqs: int = 256, clip: float = 1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    N = xs.size(0)
    pbar = trange(iters)
    for _ in pbar:
        idx = torch.randint(0, N, (batch_seqs,), device=xs.device)
        x_b, t_b = xs[idx], ts[idx]
        lpx = model.log_prob_paths(x_b, t_b)      # (B,)
        loss = -lpx.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        pbar.set_description(f"Path NLL: {loss.item():.4f}")

@torch.no_grad()
def save_samples_plot(samples: Tensor, filename: str = "plot.jpg"):
    """
    samples: (B, T, 3)
    """
    from matplotlib import cm
    B, T, D = samples.shape
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.viridis(torch.linspace(0, 1, B).cpu().numpy())
    for i in range(B):
        xi = samples[i].detach().cpu().numpy()
        ax.plot(xi[:, 0], xi[:, 1], xi[:, 2], color=colors[i], linewidth=1.5, alpha=0.95)
    ax.set_xlabel('$x_1$', fontsize=12); ax.set_ylabel('$x_2$', fontsize=12); ax.set_zlabel('$x_3$', fontsize=12)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_title("NF-pushed SDE: generated trajectories", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    device = "cuda"
    torch.manual_seed(SEED)

    # Data
    batch_size = 2 ** 10
    ts0, tf0 = 0.0, 1.0
    n_steps = 40
    noise_std = 0.01
    xs, ts = gen_data(batch_size, ts0, tf0, n_steps, noise_std, device=device)
    xs, ts = xs.to(device), ts.to(device)

    # Model
    D = xs.shape[-1]
    model = NF_SDE_Model(D=D, hidden=160, n_layers=6, tdim=64).to(device)

    train_nf_sde(model, xs, ts, iters=2000, lr=2e-3, batch_seqs=256, clip=1.0)

    us = ts[0, :, :].to(device)          # (T,1) shared grid
    n_paths = 24
    xs_gen = model.sample_paths(us, n_paths=n_paths)  # (n_paths, T, 3)
    save_samples_plot(xs_gen, "plot.jpg")

    print("Wrote generated trajectories to plot.jpg")
