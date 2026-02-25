# train.py

# Zara by Nexara - Training Script

# 

# Usage:

# python train.py –data your_file.txt

# python train.py –data_train processed/train.bin –data_val processed/val.bin

# 

# Requirements:

# pip install torch tiktoken numpy

import os
import math
import time
import argparse
import numpy as np
import torch

from model import ModelConfig, TransformerLM

def parse_args():
parser = argparse.ArgumentParser(description=“Train Zara by Nexara”)

```
parser.add_argument("--data", type=str, default=None)
parser.add_argument("--data_train", type=str, default=None)
parser.add_argument("--data_val", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="checkpoints")

parser.add_argument("--context_len", type=int, default=512)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--n_layers", type=int, default=8)
parser.add_argument("--d_ff", type=int, default=2048)
parser.add_argument("--dropout", type=float, default=0.1)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--min_lr", type=float, default=1e-5)
parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--warmup_steps", type=int, default=200)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--val_frac", type=float, default=0.05)

parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--save_every", type=int, default=1000)
parser.add_argument("--sample_every", type=int, default=500)

parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--compile", action="store_true")

return parser.parse_args()
```

def load_text_data(path, val_frac):
try:
import tiktoken
except ImportError:
raise ImportError(“Run: pip install tiktoken”)

```
enc = tiktoken.get_encoding("gpt2")

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

print("Loaded " + str(len(text)) + " characters from " + path)
tokens = enc.encode(text)
print("Tokenized to " + str(len(tokens)) + " tokens")

tokens = torch.tensor(tokens, dtype=torch.long)
split = int(len(tokens) * (1 - val_frac))
return tokens[:split], tokens[split:], enc.n_vocab, enc
```

def load_bin_data(train_path, val_path):
train_np = np.fromfile(train_path, dtype=np.uint16)
val_np = np.fromfile(val_path, dtype=np.uint16)

```
train = torch.from_numpy(train_np.astype(np.int64))
val = torch.from_numpy(val_np.astype(np.int64))

print("Train tokens: " + str(len(train)))
print("Val tokens: " + str(len(val)))

try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
except ImportError:
    enc = None
    vocab_size = 50257

return train, val, vocab_size, enc
```

def get_batch(data, batch_size, context_len, device):
ix = torch.randint(len(data) - context_len, (batch_size,))
x = torch.stack([data[i: i + context_len] for i in ix])
y = torch.stack([data[i + 1: i + context_len + 1] for i in ix])
return x.to(device), y.to(device)

def get_lr(step, warmup_steps, max_steps, lr, min_lr):
if step < warmup_steps:
return lr * step / max(warmup_steps, 1)
if step > max_steps:
return min_lr
progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
return min_lr + coeff * (lr - min_lr)

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

SAMPLE_PROMPTS = [
“# How to integrate M-Pesa STK Push in Python\n”,
“# Paystack payment integration in Django\n”,
“# USSD menu implementation\n”,
“# Flutterwave API integration\n”,
“def mpesa_stk_push(”,
]

@torch.no_grad()
def sample_code(model, enc, device, temperature=0.7, top_k=40, max_new_tokens=120):
if enc is None:
return “(sampling not available)”

```
import random
prompt = random.choice(SAMPLE_PROMPTS)

model.eval()
tokens = enc.encode(prompt)
idx = torch.tensor([tokens], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
text = enc.decode(out[0].tolist())
model.train()
return text
```

def main():
args = parse_args()
torch.manual_seed(args.seed)

```
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Zara by Nexara - Training")
print("Device: " + device)

os.makedirs(args.out_dir, exist_ok=True)

if args.data_train is not None:
    if args.data_val is None:
        raise ValueError("--data_val required when using --data_train")
    train_data, val_data, vocab_size, enc = load_bin_data(args.data_train, args.data_val)
elif args.data is not None:
    train_data, val_data, vocab_size, enc = load_text_data(args.data, args.val_frac)
else:
    raise ValueError("Provide --data or --data_train + --data_val")

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
print("Parameters: " + str(model.num_parameters()))

if args.compile:
    print("Compiling model...")
    model = torch.compile(model)

decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=args.lr,
    betas=(0.9, 0.95),
)

start_step = 0
if args.resume:
    print("Resuming from: " + args.resume)
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]
    print("Resumed from step " + str(start_step))

model.train()
best_val_loss = float("inf")
running_loss = 0.0
t0 = time.time()

print("Starting training for " + str(args.max_steps) + " steps...")
print("-" * 60)

for step in range(start_step, args.max_steps):

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

    if step % args.log_every == 0 and step > 0:
        avg_loss = running_loss / args.log_every
        elapsed = time.time() - t0
        tok_per_sec = step * args.batch_size * args.context_len / max(elapsed, 1)
        print(
            "Step " + str(step).rjust(6) +
            " | loss " + str(round(avg_loss, 4)) +
            " | lr " + "{:.2e}".format(lr_now) +
            " | " + str(int(tok_per_sec)) + " tok/s"
        )
        running_loss = 0.0

    if step > 0 and step % args.eval_every == 0:
        metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.context_len, device)
        print(
            "[Eval step " + str(step) + "] " +
            "train=" + str(round(metrics["train"], 4)) +
            " val=" + str(round(metrics["val"], 4))
        )
        if metrics["val"] < best_val_loss:
            best_val_loss = metrics["val"]
            best_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}, best_path)
            print("Best model saved -> " + best_path)

    if step > 0 and step % args.sample_every == 0:
        sample = sample_code(model, enc, device)
        print("\n--- Zara Sample at step " + str(step) + " ---")
        print(sample)
        print("-" * 60 + "\n")

    if step > 0 and step % args.save_every == 0:
        ckpt_path = os.path.join(args.out_dir, "ckpt_step" + str(step) + ".pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg.__dict__,
            "step": step,
        }, ckpt_path)
        print("Checkpoint saved -> " + ckpt_path)

final_path = os.path.join(args.out_dir, "zara_final.pt")
torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": args.max_steps}, final_path)
print("Training complete! Final model -> " + final_path)
```

if **name** == “**main**”:
main()
