## “””
train.py — Training Script for AfriCode LM

Trains the African Code Assistant on coding datasets and
African API documentation (M-Pesa, Paystack, Flutterwave, etc.)

Supports:

- Raw .txt files (tokenized on the fly)
- Pre-tokenized .bin files from data_pipeline.py (recommended)

Usage:
# From text file
python train.py –data your_code_data.txt

```
# From pre-tokenized binary files (recommended for large datasets)
python train.py --data_train processed/africode_train.bin --data_val processed/africode_val.bin
```

Requirements:
pip install torch tiktoken numpy
“””

import os
import math
import time
import argparse
import numpy as np
import torch

from model import ModelConfig, TransformerLM

# —————————————————————————

# Argument Parsing

# —————————————————————————

def parse_args():
parser = argparse.ArgumentParser(description=“Train AfriCode LM”)

```
# Data — use one of these two options
parser.add_argument("--data",       type=str, default=None,
                    help="Raw .txt file (tokenized on the fly)")
parser.add_argument("--data_train", type=str, default=None,
                    help="Pre-tokenized train .bin file (from data_pipeline.py)")
parser.add_argument("--data_val",   type=str, default=None,
                    help="Pre-tokenized val .bin file (from data_pipeline.py)")

# Output
parser.add_argument("--out_dir",      type=str,   default="checkpoints")

# Model architecture — tuned for code generation
parser.add_argument("--context_len",  type=int,   default=512,   help="Max sequence length")
parser.add_argument("--d_model",      type=int,   default=512,   help="Embedding dimension")
parser.add_argument("--n_heads",      type=int,   default=8,     help="Attention heads")
parser.add_argument("--n_layers",     type=int,   default=8,     help="Transformer layers")
parser.add_argument("--d_ff",         type=int,   default=2048,  help="Feed-forward dimension")
parser.add_argument("--dropout",      type=float, default=0.1)

# Training hyperparameters
parser.add_argument("--batch_size",   type=int,   default=16,    help="Reduce if GPU runs out of memory")
parser.add_argument("--lr",           type=float, default=3e-4,  help="Peak learning rate")
parser.add_argument("--min_lr",       type=float, default=1e-5,  help="Minimum LR after decay")
parser.add_argument("--max_steps",    type=int,   default=10000, help="Total training steps")
parser.add_argument("--warmup_steps", type=int,   default=200,   help="LR warmup steps")
parser.add_argument("--grad_clip",    type=float, default=1.0,   help="Gradient clipping")
parser.add_argument("--val_frac",     type=float, default=0.05,  help="Validation split fraction")

# Logging / checkpointing
parser.add_argument("--log_every",    type=int,   default=50)
parser.add_argument("--eval_every",   type=int,   default=500)
parser.add_argument("--save_every",   type=int,   default=1000)
parser.add_argument("--sample_every", type=int,   default=500)

# Misc
parser.add_argument("--resume",       type=str,   default=None,  help="Path to checkpoint to resume from")
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--compile",      action="store_true",        help="Use torch.compile() — PyTorch 2.0+")

return parser.parse_args()
```

# —————————————————————————

# Data Loading

# —————————————————————————

def load_text_data(path: str, val_frac: float):
“”“Tokenize a raw .txt file using GPT-2 tokenizer.”””
try:
import tiktoken
except ImportError:
raise ImportError(“Run: pip install tiktoken”)

```
enc = tiktoken.get_encoding("gpt2")

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded {len(text):,} characters from '{path}'")
tokens = enc.encode(text)
print(f"Tokenized to {len(tokens):,} tokens")

tokens = torch.tensor(tokens, dtype=torch.long)
split  = int(len(tokens) * (1 - val_frac))
return tokens[:split], tokens[split:], enc.n_vocab, enc
```

def load_bin_data(train_path: str, val_path: str):
“”“Load pre-tokenized binary files produced by data_pipeline.py.”””
train_np = np.fromfile(train_path, dtype=np.uint16)
val_np   = np.fromfile(val_path,   dtype=np.uint16)

```
train = torch.from_numpy(train_np.astype(np.int64))
val   = torch.from_numpy(val_np.astype(np.int64))

print(f"Loaded {len(train):,} train tokens | {len(val):,} val tokens")

try:
    import tiktoken
    enc        = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
except ImportError:
    enc        = None
    vocab_size = 50257

return train, val, vocab_size, enc
```

def get_batch(data: torch.Tensor, batch_size: int, context_len: int, device: str):
“”“Sample a random batch of (inputs, targets).”””
ix = torch.randint(len(data) - context_len, (batch_size,))
x  = torch.stack([data[i : i + context_len] for i in ix])
y  = torch.stack([data[i + 1 : i + context_len + 1] for i in ix])
return x.to(device), y.to(device)

# —————————————————————————

# Learning Rate Schedule — Linear Warmup + Cosine Decay

# —————————————————————————

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
# Linear warmup
if step < warmup_steps:
return lr * step / max(warmup_steps, 1)
# Minimum LR after training ends
if step > max_steps:
return min_lr
# Cosine decay
progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
return min_lr + coeff * (lr - min_lr)

# —————————————————————————

# Evaluation

# —————————————————————————

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_len, device, n_batches=20):
model.eval()
results = {}
for name, data in [(“train”, train_data), (“val”, val_data)]:
losses = [
model(*[t for t in [get_batch(data, batch_size, context_len, device)]][0:1],
get_batch(data, batch_size, context_len, device)[1])[1].item()
for _ in range(n_batches)
]
results[name] = sum(losses) / len(losses)
model.train()
return results

# —————————————————————————

# Code Sampling — African API focused prompts

# —————————————————————————

AFRICODE_PROMPTS = [
“# How to integrate M-Pesa STK Push in Python\n”,
“# Paystack payment integration in Django\n”,
“# USSD menu implementation in Python\n”,
“# Flutterwave API integration example\n”,
“# MTN Mobile Money API in JavaScript\n”,
“def mpesa_stk_push(”,
“const paystack = require(‘paystack’)(”,
]

@torch.no_grad()
def sample_code(model, enc, device, temperature=0.7, top_k=40, max_new_tokens=150):
“”“Generate a code sample using an African API prompt.”””
if enc is None:
return “(sampling unavailable — tiktoken not installed)”

```
import random
prompt = random.choice(AFRICODE_PROMPTS)

model.eval()
tokens = enc.encode(prompt)
idx    = torch.tensor([tokens], dtype=torch.long, device=device)
out    = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
text   = enc.decode(out[0].tolist())
model.train()
return text
```

# —————————————————————————

# Main Training Loop

# —————————————————————————

def main():
args = parse_args()
torch.manual_seed(args.seed)

```
# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"AfriCode LM Training | Device: {device}")

os.makedirs(args.out_dir, exist_ok=True)

# Load data
if args.data_train is not None:
    if args.data_val is None:
        raise ValueError("--data_val is required when using --data_train")
    train_data, val_data, vocab_size, enc = load_bin_data(args.data_train, args.data_val)
elif args.data is not None:
    train_data, val_data, vocab_size, enc = load_text_data(args.data, args.val_frac)
else:
    raise ValueError(
        "Please provide either:\n"
        "  --data your_file.txt\n"
        "  --data_train train.bin --data_val val.bin"
    )

# Model
cfg = ModelConfig(
    vocab_size     = vocab_size,
    context_length = args.context_len,
    d_model        = args.d_model,
    n_heads        = args.n_heads,
    n_layers       = args.n_layers,
    d_ff           = args.d_ff,
    dropout        = args.dropout,
)
model = TransformerLM(cfg).to(device)
print(f"Parameters: {model.num_parameters():,}")

if args.compile:
    print("Compiling model with torch.compile() for faster training...")
    model = torch.compile(model)

# Optimizer — separate weight decay for weights vs biases/norms
decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params,   "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=args.lr,
    betas=(0.9, 0.95),
)

# Resume from checkpoint
start_step = 0
if args.resume:
    print(f"Resuming from checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]
    print(f"Resumed from step {start_step}")

# Training
model.train()
best_val_loss = float("inf")
running_loss  = 0.0
t0            = time.time()

print(f"\nStarting AfriCode LM training for {args.max_steps} steps...\n")
print(f"{'Step':>6} | {'Loss':>8} | {'LR':>10} | {'Tok/s':>10}")
print("-" * 45)

for step in range(start_step, args.max_steps):

    # Update LR
    lr_now = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_now

    # Forward + backward
    x, y = get_batch(train_data, args.batch_size, args.context_len, device)
    _, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    running_loss += loss.item()

    # Log
    if step % args.log_every == 0 and step > 0:
        avg_loss      = running_loss / args.log_every
        elapsed       = time.time() - t0
        tokens_per_sec = step * args.batch_size * args.context_len / max(elapsed, 1)
        print(f"{step:>6} | {avg_loss:>8.4f} | {lr_now:>10.2e} | {tokens_per_sec:>10,.0f}")
        running_loss = 0.0

    # Evaluate
    if step > 0 and step % args.eval_every == 0:
        metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.context_len, device)
        print(f"\n  [Eval @ step {step}] train_loss={metrics['train']:.4f}  val_loss={metrics['val']:.4f}")

        if metrics["val"] < best_val_loss:
            best_val_loss = metrics["val"]
            best_path     = os.path.join(args.out_dir, "best_model.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}, best_path)
            print(f"  ✓ Best model saved → {best_path}\n")

    # Sample code
    if step > 0 and step % args.sample_every == 0:
        sample = sample_code(model, enc, device)
        print(f"\n{'='*50}")
        print(f"AfriCode Sample @ step {step}:")
        print(sample)
        print(f"{'='*50}\n")

    # Save checkpoint
    if step > 0 and step % args.save_every == 0:
        ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step}.pt")
        torch.save({
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg":       cfg.__dict__,
            "step":      step,
        }, ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")

# Final save
final_path = os.path.join(args.out_dir, "africode_final.pt")
torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": args.max_steps}, final_path)
print(f"\nTraining complete! Final model → {final_path}")
```

if **name** == “**main**”:
main()
