from typing import TypeAlias

import jaxtyping
import torch
import torch.nn as nn
from typeguard import typechecked as typechecker

from toyllm.device import get_device
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

        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)  # Query Weight
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)  # Key Weight
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)  # Value Weight
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1))

    @jaxtyping.jaxtyped(typechecker=typechecker)
    def forward(self, x: GPTInnerType) -> GPTInnerType:
        batch_size, num_tokens, _d_in = x.shape

        # (batch_size, num_tokens, d_in) -> (batch_size, num_tokens, d_out)
        keys = self.Wk(x)
        queries = self.Wq(x)
        values = self.Wv(x)

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


def generate_text_simple(model: GPTModel, text_token_ids: torch.Tensor, max_new_tokens: int, ctx_len: int):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size(ctx_len)
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        context_text_token_ids = text_token_ids[:, -ctx_len:]

        # Get the predictions
        with torch.no_grad():
            logits = model(context_text_token_ids)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        text_token_ids = torch.cat((text_token_ids, next_token_id), dim=1)  # (batch, n_tokens+1)

    return text_token_ids


if __name__ == "__main__":
    from toyllm.model.config import GPT_CONFIG_124M
    from toyllm.model.tokenizer import (
        get_gpt2_tokenizer,
        text_to_token_ids,
        token_ids_to_text,
    )

    seed = 42
    device = get_device()

    torch.manual_seed(seed)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    model.eval()  # disable dropout
    start_context = "Hello, I am"

    tokenizer = get_gpt2_tokenizer()
    text_token_ids = text_to_token_ids(start_context).to(device)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("encoded_tensor.shape:", text_token_ids.shape)

    out = generate_text_simple(
        model=model,
        text_token_ids=text_token_ids,
        max_new_tokens=10,
        ctx_len=GPT_CONFIG_124M.ctx_len,
    )
    decoded_text = token_ids_to_text(out)

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
