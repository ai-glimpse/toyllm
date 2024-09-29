# Copyright 2024 Xiangzhuang Shen
# Copyright 2023-2024 Sebastian Raschka
# SPDX-License-Identifier: Apache-2.0


from typing import TypeAlias

import jaxtyping
import torch
import torch.nn as nn
from typeguard import typechecked as typechecker

from toyllm.model.config import GPTModelConfig

GPTInputType: TypeAlias = jaxtyping.Int[torch.Tensor, "batch_size num_tokens"]
GPTInnerType: TypeAlias = jaxtyping.Float[torch.Tensor, "batch_size num_tokens emb_dim"]
GPTOutputType: TypeAlias = jaxtyping.Float[torch.Tensor, "batch_size num_tokens vocab_size"]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        ctx_len: int,
        dropout_rate: float,
        n_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Query Weight
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # Key Weight
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # Value Weight
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1))

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def forward(self, x: GPTInnerType) -> GPTInnerType:
        batch_size, num_tokens, _d_in = x.shape

        # (batch_size, num_tokens, d_in) -> (batch_size, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, head_dim)
        keys = keys.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.n_heads, self.head_dim)

        # Transpose: (batch_size, num_tokens, num_heads, head_dim) -> (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # attn_scores shape: (batch_size, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # TODO: explain why dropout here
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def forward(self, x: GPTInnerType) -> GPTInnerType:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def forward(self, x: GPTInnerType) -> GPTInnerType:
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTModelConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
            nn.Dropout(cfg.drop_rate),
        )

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def forward(self, x: GPTInnerType) -> GPTInnerType:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTModelConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            ctx_len=cfg.ctx_len,
            n_heads=cfg.n_heads,
            dropout_rate=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_resid = nn.Dropout(cfg.drop_rate)

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def forward(self, x: GPTInnerType) -> GPTInnerType:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTModelConfig):
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def forward(self, input_vocab_indexes: GPTInputType) -> GPTOutputType:
        _batch_size, num_tokens = input_vocab_indexes.shape
        # batch_size num_tokens -> batch_size num_tokens emb_dim
        tok_embeds = self.tok_emb(input_vocab_indexes)
        # pos_embeds shape: (num_tokens, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(num_tokens, device=input_vocab_indexes.device))
        # pos_embeds is **broadcast** to (batch_size, num_tokens, emb_dim)
        # x: (batch_size, num_tokens, emb_dim)
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def save(self):
        torch.save(self.state_dict(), f"{self.config.name}.pt")
    
    def load(self, model_path: str):
        self.load_state_dict(torch.load(model_path, weights_only=True))
