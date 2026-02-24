## “””
train.py — Training script for TransformerLM

Supports both:

- Raw .txt files (tokenized on the fly)
- Pre-tokenized .bin files (from data_pipeline.py — much faster)

Usage:
# From text file
python train.py –data your_text.txt

```
# From pre-tokenized binary files (recommended for large datasets)
python train.py --data_train processed/data_train.bin --data_val processed/data_val.bin
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
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import ModelConfig, TransformerLM

# —————————————————————————

# Argument parsing

# —————————————————————————

def parse_args():
parser = argparse.ArgumentParser(description=“Train TransformerLM”)

```
# Data sources
data_group = parser.add_mutually_exclusive_group()
data_group.add_argument("--data",       type=str, help="Raw .txt file (tokenized on the fly)")
data_group.add_argument("--data_train", type=str, help="Pre-tokenized train .bin file")
parser.add_argument("--data_val",       type=str, help="Pre-tokenized val .bin file")

# Output
parser.add_argument("--out_dir",
                    type=str,
                    default="/content/drive/MyDrive/ai_checkpoints/")

# Model architecture
parser.add_argument("--context_len",  type=int,   default=256)
parser.add_argument("--d_model",      type=int,   default=384)
parser.add_argument("--n_heads",      type=int,   default=6)
parser.add_argument("--n_layers",     type=int,   default=6)
parser.add_argument("--d_ff",         type=int,   default=1536)
parser.add_argument("--dropout",      type=float, default=0.1)

# Training
parser.add_argument("--batch_size",   type=int,   default=32)
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--min_lr",       type=float, default=1e-5)
parser.add_argument("--max_steps",    type=int,   default=5000)
parser.add_argument("--warmup_steps", type=int,   default=100)
parser.add_argument("--grad_clip",    type=float, default=1.0)
parser.add_argument("--val_frac",     type=float, default=0.1)

# Logging / saving
parser.add_argument("--eval_every",   type=int,   default=500)
parser.add_argument("--save_every",   type=int,   default=1000)
parser.add_argument("--sample_every", type=int,   default=500)
parser.add_argument("--log_every",    type=int,   default=50)

# Misc
parser.add_argument("--resume",       type=str,   default=None)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--compile",      action="store_true", help="Use torch.compile() for speedup (PyTorch 2.0+)")

return parser.parse_args()
```

# —————————————————————————

# Data loading

# —————————————————————————

def load_text_data(path: str, val_frac: float):
“”“Load and tokenize a raw text file.”””
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
split = int(len(tokens) * (1 - val_frac))
return tokens[:split], tokens[split:], enc.n_vocab, enc
```

def load_bin_data(train_path: str, val_path: str):
“”“Load pre-tokenized binary files from data_pipeline.py.”””
train = np.fromfile(train_path, dtype=np.uint16)
val   = np.fromfile(val_path,   dtype=np.uint16)

```
train = torch.from_numpy(train.astype(np.int64))
val   = torch.from_numpy(val.astype(np.int64))

print(f"Loaded {len(train):,} train tokens from '{train_path}'")
print(f"Loaded {len(val):,} val tokens from '{val_path}'")

try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
except ImportError:
    vocab_size = 50257
    enc = None

return train, val, vocab_size, enc
```

def get_batch(data: torch.Tensor, batch_size: int, context_len: int, device: str):
“”“Sample a random batch.”””
ix = torch.randint(len(data) - context_len, (batch_size,))
x = torch.stack([data[i : i + context_len] for i in ix])
y = torch.stack([data[i + 1 : i + context_len + 1] for i in ix])
return x.to(device), y.to(device)

# —————————————————————————

# Learning rate schedule (linear warmup + cosine decay)

# —————————————————————————

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
if step < warmup_steps:
return lr * step / warmup_steps
if step > max_steps:
return min_lr
decay = (step - warmup_steps) / (max_steps - warmup_steps)
coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
return min_lr + coeff * (lr - min_lr)

# —————————————————————————

# Evaluation

# —————————————————————————

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_len, device, n_batches=20):
model.eval()
results = {}
for name, data in [(“train”, train_data), (“val”, val_data)]:
losses = []
for _ in range(n_batches):
x, y = get_batch(data, batch_size, context_len, device)
_, loss = model(x, y)
losses.append(loss.item())
results[name] = sum(losses) / len(losses)
model.train()
return results

# —————————————————————————

# Text sampling

# —————————————————————————

@torch.no_grad()
def sample_text(model, enc, device, max_new_tokens=150, temperature=0.8, top_k=50):
if enc is None:
return “(sampling unavailable — tiktoken not installed)”
model.eval()
prompt = “The “
tokens = enc.encode(prompt)
idx = torch.tensor([tokens], dtype=torch.long, device=device)
generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
text = enc.decode(generated[0].tolist())
model.train()
return text

# —————————————————————————

# Main

# —————————————————————————

def main():
args = parse_args()

os.makedirs(args.out_dir, exist_ok=True)
torch.manual_seed(args.seed)

```
# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")

os.makedirs(args.out_dir, exist_ok=True)

# Load data
if args.data:
    train_data, val_data, vocab_size, enc = load_text_data(args.data, args.val_frac)
elif args.data_train:
    if not args.data_val:
        raise ValueError("--data_val is required when using --data_train")
    train_data, val_data, vocab_size, enc = load_bin_data(args.data_train, args.data_val)
else:
    raise ValueError("Provide either --data (txt) or --data_train + --data_val (bin)")

print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

# Model
cfg = ModelConfig(
    vocab_size=vocab_size,
    context_length=args.context_len,
    d_model=args.d_model,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    d_ff=args.d_ff,
    dropout=args.dropout,
)
model = TransformerLM(cfg).to(device)

if args.compile:
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

print(f"Parameters: {model.num_parameters():,}")

# Optimizer (no weight decay on biases and layer norms)
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay_params,   "weight_decay": 0.1},
    {"params": nodecay_params, "weight_decay": 0.0},
], lr=args.lr, betas=(0.9, 0.95))

# Resume from checkpoint
start_step = 0
if args.resume:
    print(f"Resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]

# Training loop
model.train()
best_val_loss = float("inf")
t0 = time.time()
running_loss = 0.0

print(f"\nStarting training for {args.max_steps} steps...\n")

for step in range(start_step, args.max_steps):

    # Manual LR schedule
    lr_now = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_now

    x, y = get_batch(train_data, args.batch_size, args.context_len, device)
    _, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    running_loss += loss.item()

    # Log
    if step % args.log_every == 0:
        avg_loss = running_loss / max(args.log_every, 1)
        elapsed = time.time() - t0
        tokens_per_sec = (step + 1) * args.batch_size * args.context_len / elapsed
        print(f"Step {step:5d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | {tokens_per_sec:,.0f} tok/s")
        running_loss = 0.0

    # Evaluate
    if step > 0 and step % args.eval_every == 0:
        metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.context_len, device)
        print(f"\n  [Eval step {step}] train={metrics['train']:.4f}  val={metrics['val']:.4f}")

        if metrics["val"] < best_val_loss:
            best_val_loss = metrics["val"]
            best_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}, best_path)
            print(f"  ✓ New best val loss! Saved to {best_path}\n")

    # Sample text
    if step > 0 and step % args.sample_every == 0:
        sample = sample_text(model, enc, device)
        print(f"\n--- Sample at step {step} ---\n{sample}\n{'='*40}\n")

    # Save checkpoint
    if step > 0 and step % args.save_every == 0:
        ckpt_path = f"/content/drive/MyDrive/ai_checkpoints/ckpt_step{step}.pt")
        torch.save({
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg":       cfg.__dict__,
            "step":      step,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

# Final save
final_path = os.path.join(args.out_dir, "final_model.pt")
torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": args.max_steps}, final_path)
print(f"\nTraining complete! Final model saved to {final_path}")
```

if **name** == “**main**”:
main()
