"""Attention-only byte-level GPT with the residual stream projected onto a polytope.

Every arm has the same structure. Attention reads through a pre-norm (LayerNorm),
and after each add the stream is renormalized and projected:

    x <- P(RMSNorm(x + attn(LN(x))))

The same P(RMSNorm(.)) is applied once to the embeddings; then a final LayerNorm
and the (tied) unembedding. The RMSNorm has no gain, so the model cannot shrink
the stream inside P to dodge the projection.

Arms (P from polytope.proj):
    floor    P = identity
    box      P = clamp(-1, 1)
    l1       P = projection onto the L1 ball of radius d/4
    shrink   P = soft-threshold at a fixed 0.78
    ceiling  floor plus a GELU MLP sublayer: x <- RMSNorm(x + mlp(LN(x)))
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .proj import PROJ_ARMS, NormProj

ARMS = list(PROJ_ARMS)


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    mlp_ratio: int = 4                # ceiling only
    dropout: float = 0.0
    arm: str = "floor"


class GeluMLP(nn.Module):
    def __init__(self, d, m, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(d, m)
        self.proj = nn.Linear(m, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(cfg)
        self.p1 = NormProj(PROJ_ARMS[cfg.arm], d)
        self.mlp = None
        if cfg.arm == "ceiling":
            self.ln2 = nn.LayerNorm(d)
            self.mlp = GeluMLP(d, cfg.mlp_ratio * d, cfg.dropout)
            self.p2 = NormProj("none", d)

    def forward(self, x):
        x = self.p1(x + self.attn(self.ln1(x)))
        if self.mlp is not None:
            x = self.p2(x + self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.arm not in ARMS:
            raise ValueError(f"unknown arm {cfg.arm!r}; choose from {ARMS}")
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.p_emb = NormProj(PROJ_ARMS[cfg.arm], cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        for n, p in self.named_parameters():   # GPT-2 scaled init on residual projections
            if n.endswith("proj.weight") and p.dim() == 2:
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(mod):
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0.0, 0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, 0.0, 0.02)

    def proj_layers(self):
        """(name, NormProj) for the projections: 'emb', then 'L0'..'L{n-1}' after each attention."""
        return [("emb", self.p_emb)] + [(f"L{i}", b.p1) for i, b in enumerate(self.blocks)]

    def set_tracking(self, on: bool):
        for _, m in self.proj_layers():
            m.track_stats = on

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.p_emb(self.drop(self.tok(idx) + self.pos(pos)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
        return logits, loss


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
