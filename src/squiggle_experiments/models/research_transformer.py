"""
Research-grade transformer with modern LLM architecture.

Features:
- Decoder-only causal attention
- RMSNorm (pre-norm architecture)
- Rotary Position Embeddings (RoPE)
- SwiGLU activation
- Tied input/output embeddings
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ResearchTransformerConfig:
    """Configuration for research-grade transformer."""

    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: Optional[int] = None  # For GQA; None = MHA (n_kv_heads = n_heads)
    d_ff: Optional[int] = None  # If None, computed as int(2.75 * d_model)
    rope_base: float = 10000.0
    rope_scaling: Optional[float] = None  # For extended context
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0
    bias: bool = False  # Most modern LLMs don't use bias

    def __post_init__(self):
        # Validate
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.n_kv_heads is not None:
            assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def effective_n_kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def effective_d_ff(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        # SwiGLU typically uses 2.75x multiplier (rounded to multiple of 256 for efficiency)
        raw = int(2.75 * self.d_model)
        return ((raw + 255) // 256) * 256

    def param_count(self) -> int:
        """Estimate total parameter count."""
        d = self.d_model
        h = self.n_heads
        kv = self.effective_n_kv_heads
        ff = self.effective_d_ff
        v = self.vocab_size
        L = self.n_layers

        # Embeddings (tied, so count once)
        embed = v * d

        # Per layer:
        # - Attention: Q, K, V projections + output projection
        #   Q: d * d, K: d * (d * kv/h), V: d * (d * kv/h), O: d * d
        attn_qo = 2 * d * d
        attn_kv = 2 * d * (d * kv // h)
        attn = attn_qo + attn_kv

        # - MLP (SwiGLU): gate, up, down
        #   gate: d * ff, up: d * ff, down: ff * d
        mlp = 3 * d * ff

        # - RMSNorm: 2 per layer (attn + mlp)
        norm = 2 * d

        per_layer = attn + mlp + norm

        # Final norm
        final_norm = d

        total = embed + L * per_layer + final_norm
        return total


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    scaling: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute rotary embedding frequencies."""
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Apply scaling if specified
    if scaling is not None:
        inv_freq = inv_freq / scaling

    # Compute position indices
    t = torch.arange(max_seq_len, device=device).float()

    # Outer product: (seq_len, dim/2)
    freqs = torch.outer(t, inv_freq)

    # Complex exponentials for rotation
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i * freq)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    # xq, xk: (batch, seq_len, n_heads, head_dim)
    # freqs_cis: (seq_len, head_dim/2)

    # Reshape for complex multiplication
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)  # (..., head_dim/2, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Convert to complex
    xq_c = torch.view_as_complex(xq_r)  # (..., head_dim/2)
    xk_c = torch.view_as_complex(xk_r)

    # Get relevant frequencies
    seq_len = xq.size(1)
    freqs = freqs_cis[:seq_len]

    # Reshape freqs for broadcasting: (1, seq_len, 1, head_dim/2)
    freqs = freqs.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish(xW_gate) * (xW_up)."""

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional GQA."""

    def __init__(self, cfg: ResearchTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.effective_n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # For GQA key/value repetition

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=cfg.bias)

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Repeat K, V for GQA if needed
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, layer_idx: int, cfg: ResearchTransformerConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg

        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.effective_d_ff, bias=cfg.bias, dropout=cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        h = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        # Pre-norm MLP
        out = h + self.mlp(self.mlp_norm(h))
        return out


class ResearchTransformerLM(nn.Module):
    """
    Research-grade decoder-only transformer language model.

    Modern LLM architecture with RMSNorm, RoPE, SwiGLU, and tied embeddings.
    """

    def __init__(self, cfg: ResearchTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Token embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(i, cfg) for i in range(cfg.n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

        # Output projection (tied with embeddings if configured)
        if cfg.tie_embeddings:
            self.lm_head = None  # Use tok_emb.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                cfg.head_dim,
                cfg.max_seq_len,
                cfg.rope_base,
                cfg.rope_scaling,
            ),
            persistent=False,
        )

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")), diagonal=1),
            persistent=False,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) token indices
            targets: (batch, seq_len) target token indices for loss computation

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar loss if targets provided, else None
        """
        batch, seq_len = input_ids.shape
        assert seq_len <= self.cfg.max_seq_len, f"seq_len {seq_len} > max {self.cfg.max_seq_len}"

        # Token embeddings
        x = self.tok_emb(input_ids)

        # Get relevant portion of precomputed values
        freqs_cis = self.freqs_cis[:seq_len]
        mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer layers
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        # Final norm
        x = self.norm(x)

        # Output logits
        if self.cfg.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_layer_output(self, layer_idx: int) -> nn.Module:
        """Get a specific transformer layer for instrumentation."""
        return self.layers[layer_idx]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids[:, -self.cfg.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Preset configurations
def get_research_config_350m() -> ResearchTransformerConfig:
    """~350M parameter configuration for fast iteration."""
    return ResearchTransformerConfig(
        vocab_size=32000,
        max_seq_len=2048,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=2816,  # ~2.75x
        tie_embeddings=True,
    )


def get_research_config_1b() -> ResearchTransformerConfig:
    """~1.3B parameter configuration for research legitimacy."""
    return ResearchTransformerConfig(
        vocab_size=32000,
        max_seq_len=2048,
        d_model=2048,
        n_layers=24,
        n_heads=16,
        d_ff=5632,  # ~2.75x
        tie_embeddings=True,
    )


def get_research_config_debug() -> ResearchTransformerConfig:
    """Tiny config for debugging."""
    return ResearchTransformerConfig(
        vocab_size=1000,
        max_seq_len=256,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=704,
        tie_embeddings=True,
    )
