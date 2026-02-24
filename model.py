## “””
Transformer Language Model

A GPT-style decoder-only transformer you can train from scratch.
“””

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# —————————————————————————

# Configuration

# —————————————————————————

class ModelConfig:
def **init**(
self,
vocab_size: int = 50257,      # GPT-2 tokenizer default; change to match yours
context_length: int = 256,    # max sequence length (tokens)
d_model: int = 384,           # embedding / hidden dimension
n_heads: int = 6,             # attention heads (d_model must be divisible)
n_layers: int = 6,            # transformer blocks
d_ff: int = 1536,             # feed-forward inner dimension (usually 4 * d_model)
dropout: float = 0.1,
bias: bool = True,            # use bias in linear layers?
):
assert d_model % n_heads == 0, “d_model must be divisible by n_heads”
self.vocab_size = vocab_size
self.context_length = context_length
self.d_model = d_model
self.n_heads = n_heads
self.n_layers = n_layers
self.d_ff = d_ff
self.dropout = dropout
self.bias = bias

```
def __repr__(self):
    total = sum(p.numel() for p in TransformerLM(self).parameters())
    return (
        f"ModelConfig(vocab={self.vocab_size}, ctx={self.context_length}, "
        f"d={self.d_model}, heads={self.n_heads}, layers={self.n_layers}, "
        f"~{total/1e6:.1f}M params)"
    )
```

# —————————————————————————

# Building blocks

# —————————————————————————

class MultiHeadAttention(nn.Module):
“”“Causal (masked) multi-head self-attention.”””

```
def __init__(self, cfg: ModelConfig):
    super().__init__()
    self.n_heads = cfg.n_heads
    self.d_head = cfg.d_model // cfg.n_heads

    self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
    self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
    self.attn_drop = nn.Dropout(cfg.dropout)
    self.resid_drop = nn.Dropout(cfg.dropout)

    # Causal mask (lower-triangular)
    mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length))
    self.register_buffer("mask", mask.view(1, 1, cfg.context_length, cfg.context_length))

def forward(self, x):
    B, T, C = x.shape

    # Project to Q, K, V
    q, k, v = self.qkv(x).split(C, dim=-1)

    # Reshape to (B, heads, T, d_head)
    def reshape(t):
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    q, k, v = reshape(q), reshape(k), reshape(v)

    # Scaled dot-product attention
    scale = math.sqrt(self.d_head)
    attn = (q @ k.transpose(-2, -1)) / scale          # (B, H, T, T)
    attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    attn = self.attn_drop(attn)

    out = attn @ v                                      # (B, H, T, d_head)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_drop(self.out_proj(out))
```

class FeedForward(nn.Module):
“”“Position-wise feed-forward network with GELU activation.”””

```
def __init__(self, cfg: ModelConfig):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias),
        nn.GELU(),
        nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias),
        nn.Dropout(cfg.dropout),
    )

def forward(self, x):
    return self.net(x)
```

class TransformerBlock(nn.Module):
“”“Pre-norm transformer block: LayerNorm → Attention → residual,
LayerNorm → FFN → residual.”””

```
def __init__(self, cfg: ModelConfig):
    super().__init__()
    self.ln1 = nn.LayerNorm(cfg.d_model)
    self.attn = MultiHeadAttention(cfg)
    self.ln2 = nn.LayerNorm(cfg.d_model)
    self.ff = FeedForward(cfg)

def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ff(self.ln2(x))
    return x
```

# —————————————————————————

# Full model

# —————————————————————————

class TransformerLM(nn.Module):
“”“GPT-style causal language model.”””

```
def __init__(self, cfg: ModelConfig):
    super().__init__()
    self.cfg = cfg

    self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
    self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
    self.drop = nn.Dropout(cfg.dropout)
    self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
    self.ln_f = nn.LayerNorm(cfg.d_model)
    self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    # Weight tying: share token embedding and output projection weights
    self.lm_head.weight = self.tok_emb.weight

    self._init_weights()

def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

def forward(self, idx, targets=None):
    """
    Args:
        idx:     (B, T) integer token ids
        targets: (B, T) integer token ids shifted by 1 (for next-token prediction)
    Returns:
        logits: (B, T, vocab_size)
        loss:   scalar cross-entropy loss (or None if targets not provided)
    """
    B, T = idx.shape
    assert T <= self.cfg.context_length, (
        f"Sequence length {T} exceeds context_length {self.cfg.context_length}"
    )

    positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
    x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, loss

@torch.no_grad()
def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
    """
    Autoregressively generate tokens.

    Args:
        idx:            (B, T) seed token ids
        max_new_tokens: how many tokens to generate
        temperature:    >1 = more random, <1 = more focused
        top_k:          if set, restrict sampling to top-k tokens

    Returns:
        (B, T + max_new_tokens) token ids
    """
    for _ in range(max_new_tokens):
        # Crop to context window
        idx_cond = idx[:, -self.cfg.context_length:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature  # last time step

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx

def num_parameters(self, trainable_only=True):
    params = self.parameters() if not trainable_only else filter(lambda p: p.requires_grad, self.parameters())
    return sum(p.numel() for p in params)
```
