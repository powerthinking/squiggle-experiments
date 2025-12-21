from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TinyTransformerConfig:
    vocab_size: int
    seq_len: int = 4
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)

        # MLP
        h = self.ln2(x)
        x = x + self.dropout(self.mlp(h))
        return x


class TinyTransformerLM(nn.Module):
    """
    Minimal transformer that predicts a single token from the final position.
    """
    def __init__(self, cfg: TinyTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T)
        returns logits: (B, V) for the final token position
        """
        b, t = input_ids.shape
        if t != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got {t}")

        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        last = x[:, -1, :]  # (B, D)
        logits = self.lm_head(last)  # (B, V)
        return logits

    def loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(input_ids)
        return F.cross_entropy(logits, targets)
