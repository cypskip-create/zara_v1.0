## “””
generate.py — Load a trained model and generate text

Usage:
python generate.py –checkpoint checkpoints/best_model.pt –prompt “Once upon a time”
“””

import argparse
import torch
from model import ModelConfig, TransformerLM

def parse_args():
parser = argparse.ArgumentParser()
parser.add_argument(”–checkpoint”,    type=str, required=True)
parser.add_argument(”–prompt”,        type=str, default=“The “)
parser.add_argument(”–max_new_tokens”,type=int, default=200)
parser.add_argument(”–temperature”,   type=float, default=0.8)
parser.add_argument(”–top_k”,         type=int, default=50)
parser.add_argument(”–num_samples”,   type=int, default=3)
return parser.parse_args()

def main():
args = parse_args()

```
try:
    import tiktoken
except ImportError:
    raise ImportError("Run: pip install tiktoken")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load checkpoint
ckpt = torch.load(args.checkpoint, map_location=device)
cfg = ModelConfig(**ckpt["cfg"])
model = TransformerLM(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded model ({model.num_parameters():,} params) from {args.checkpoint}")

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(args.prompt)
idx = torch.tensor([tokens], dtype=torch.long, device=device)

print(f"\nPrompt: {args.prompt}\n{'='*50}")
for i in range(args.num_samples):
    generated = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = enc.decode(generated[0].tolist())
    print(f"\n--- Sample {i+1} ---\n{text}\n")
```

if **name** == “**main**”:
main()
