"""Projections P applied to the (RMS-normalized) residual stream, and their stats.

    floor   identity
    box     clamp to [-1, 1]^d                       (per coordinate; hardtanh)
    l1      exact Euclidean projection onto the L1 ball of radius r = d/4
            (sort, cumsum, one threshold tau shared by all coordinates; identity
            when the input is already inside)
    shrink  sign(z) * max(|z| - tau0, 0), fixed tau0 = 0.78 (not a polytope)

All are plain autograd functions: the L1 projection is piecewise linear and the
gradient flows through tau (which depends on the sorted values).
"""
from __future__ import annotations

import torch
import torch.nn as nn

SHRINK_TAU = 0.78


def proj_box(z: torch.Tensor) -> torch.Tensor:
    return z.clamp(-1.0, 1.0)


def shrink(z: torch.Tensor, tau: float = SHRINK_TAU) -> torch.Tensor:
    return torch.sign(z) * torch.relu(z.abs() - tau)


def l1_tau(z: torch.Tensor, r: float) -> torch.Tensor:
    """Threshold tau >= 0 with sum(relu(|z| - tau)) = r (0 when |z|_1 <= r). Shape [..., 1]."""
    a = z.abs()
    s, _ = a.sort(-1, descending=True)
    cs = s.cumsum(-1) - r
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    rho = (s * k > cs).sum(-1, keepdim=True)          # >= 1 since r > 0; condition is a prefix
    tau = cs.gather(-1, rho - 1) / rho.to(z.dtype)
    return tau.clamp_min(0.0)


def proj_l1(z: torch.Tensor, r: float) -> torch.Tensor:
    tau = l1_tau(z, r)
    out = torch.sign(z) * torch.relu(z.abs() - tau)
    inside = z.abs().sum(-1, keepdim=True) <= r
    return torch.where(inside, z, out)


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """No learnable gain on purpose: a gain could shrink the stream inside P."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


PROJ_ARMS = {"floor": "none", "ceiling": "none", "box": "box", "l1": "l1", "shrink": "shrink"}


class NormProj(nn.Module):
    """x -> P(RMSNorm(x)). With track_stats on, records per-call stats in self.stats."""

    def __init__(self, kind: str, d: int):
        super().__init__()
        assert kind in ("none", "box", "l1", "shrink"), kind
        self.kind, self.r = kind, d / 4
        self.track_stats = False
        self.stats: dict = {}

    def forward(self, x):
        z = rms_norm(x)
        if self.kind == "box":
            y = proj_box(z)
        elif self.kind == "l1":
            y = proj_l1(z, self.r)
        elif self.kind == "shrink":
            y = shrink(z)
        else:
            y = z
        if self.track_stats:
            self.stats = self._stats(z.detach(), y.detach())
        return y

    @torch.no_grad()
    def _stats(self, z, y):
        changed = (y != z)                                   # coords the projection moved
        nnz = (y != 0).float().mean(-1).flatten()            # per-token nonzero fraction
        st = dict(frac_tokens_changed=changed.any(-1).float().mean().item(),
                  frac_coords_changed=changed.float().mean().item(),
                  nonzero_mean=nnz.mean().item(),
                  nonzero_median=nnz.median().item())
        if self.kind == "l1":
            st["tau_mean"] = l1_tau(z, self.r).mean().item()
        return st
