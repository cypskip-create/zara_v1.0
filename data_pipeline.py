## “””
data_pipeline.py AfriCode LM Data Pipeline

Collects, cleans, and prepares training data specifically for the
African Code Assistant, including:

- African API documentation (M-Pesa, Paystack, Flutterwave, MTN MoMo, Airtel Money)
- Code from GitHub repositories
- HuggingFace coding datasets
- African tech blog content
- Custom Q&A pairs for fine-tuning

Usage:
# African API docs + web scraping
python data_pipeline.py –source web –urls africode_urls.txt –output africode

```
# HuggingFace coding dataset (recommended first step)
python data_pipeline.py --source hf --dataset bigcode/the-stack-smol --output africode

# Your own collected files
python data_pipeline.py --source files --input_dir ./raw_data --output africode

# Fine-tuning Q&A pairs
python data_pipeline.py --source qa --input_dir ./qa_pairs --output africode_ft
```

Requirements:
pip install tiktoken datasets requests beautifulsoup4 tqdm numpy
“””

import os
import re
import sys
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Iterator

# —————————————————————————

# African API & Tech URLs — Pre-loaded for AfriCode Training

# —————————————————————————

AFRICODE_URLS = [
# M-Pesa / Safaricom
“https://developer.safaricom.co.ke/APIs”,
“https://developer.safaricom.co.ke/Documentation”,

```
# Paystack
"https://paystack.com/docs/api/",
"https://paystack.com/docs/payments/accept-payments/",
"https://paystack.com/docs/libraries-and-plugins/",

# Flutterwave
"https://developer.flutterwave.com/docs",
"https://developer.flutterwave.com/reference",

# MTN MoMo
"https://momodeveloper.mtn.com/docs",

# African tech blogs
"https://engineering.paystack.com",
"https://medium.com/andela",
```

]

# —————————————————————————

# Argument Parsing

# —————————————————————————

def parse_args():
parser = argparse.ArgumentParser(description=“AfriCode LM Data Pipeline”)
parser.add_argument(”–source”,     type=str, required=True,
choices=[“files”, “web”, “hf”, “qa”, “africode”],
help=“Data source type:\n”
“  files    — local .txt/.json files\n”
“  web      — scrape URLs from a file\n”
“  hf       — HuggingFace dataset\n”
“  qa       — Q&A pairs for fine-tuning\n”
“  africode — auto-scrape African API docs”)
parser.add_argument(”–input_dir”,      type=str, default=”./raw_data”)
parser.add_argument(”–urls”,           type=str, default=“africode_urls.txt”)
parser.add_argument(”–dataset”,        type=str, default=“bigcode/the-stack-smol”)
parser.add_argument(”–dataset_config”, type=str, default=“python”,
help=“Language subset for code datasets (python, javascript, etc.)”)
parser.add_argument(”–output”,         type=str, default=“africode”)
parser.add_argument(”–output_dir”,     type=str, default=”./processed”)
parser.add_argument(”–min_length”,     type=int, default=50)
parser.add_argument(”–max_length”,     type=int, default=50_000)
parser.add_argument(”–val_frac”,       type=float, default=0.05)
parser.add_argument(”–analyze”,        action=“store_true”)
return parser.parse_args()

# —————————————————————————

# Text Cleaning — Code-Aware

# —————————————————————————

class TextCleaner:
“”“Cleans text while preserving code structure (indentation, syntax).”””

```
def __init__(self, min_length=50, max_length=50_000):
    self.min_length = min_length
    self.max_length = max_length

def clean(self, text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes and control characters (preserve tabs for code indentation)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse excessive blank lines (keep up to 2 for code readability)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()

def is_valid(self, text: str) -> bool:
    if len(text) < self.min_length:
        return False
    # For code: allow lower alpha ratio (code has symbols, numbers, brackets)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.1:  # very low threshold for code files
        return False
    return True

def truncate(self, text: str) -> str:
    if len(text) > self.max_length:
        return text[:self.max_length]
    return text

def process(self, text: str):
    text = self.clean(text)
    if not self.is_valid(text):
        return None
    return self.truncate(text)
```

# —————————————————————————

# Deduplication

# —————————————————————————

class Deduplicator:
def **init**(self):
self.seen = set()

```
def is_duplicate(self, text: str) -> bool:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    if h in self.seen:
        return True
    self.seen.add(h)
    return False
```

# —————————————————————————

# Data Sources

# —————————————————————————

class FileSource:
“”“Load code and text from local files.”””

```
EXTENSIONS = [".txt", ".md", ".py", ".js", ".php", ".java", ".json", ".ts", ".go"]

def __init__(self, input_dir: str):
    self.input_dir = Path(input_dir)
    self.files = []
    for ext in self.EXTENSIONS:
        self.files.extend(self.input_dir.rglob(f"*{ext}"))

    if not self.files:
        raise FileNotFoundError(f"No supported files found in {input_dir}")
    print(f"Found {len(self.files)} files in {input_dir}")

def iter_documents(self) -> Iterator[str]:
    for filepath in tqdm(self.files, desc="Reading files"):
        try:
            if filepath.suffix == ".json":
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            yield item
                        elif isinstance(item, dict):
                            for key in ["content", "text", "code", "body"]:
                                if key in item:
                                    yield str(item[key])
                                    break
                elif isinstance(data, dict):
                    for key in ["content", "text", "code", "body"]:
                        if key in data:
                            yield str(data[key])
            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    yield f.read()
        except Exception as e:
            print(f"  Warning: Could not read {filepath}: {e}")
```

class WebSource:
“”“Scrape text/code from a list of URLs.”””

```
def __init__(self, urls_file: str):
    try:
        import requests
        from bs4 import BeautifulSoup
        self.requests      = requests
        self.BeautifulSoup = BeautifulSoup
    except ImportError:
        raise ImportError("Run: pip install requests beautifulsoup4")

    with open(urls_file, "r") as f:
        self.urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    print(f"Loaded {len(self.urls)} URLs")

def iter_documents(self) -> Iterator[str]:
    for url in tqdm(self.urls, desc="Scraping URLs"):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (AfriCodeBot/1.0)"}
            resp    = self.requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            soup = self.BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "ads"]):
                tag.decompose()

            # Extract code blocks specifically
            code_blocks = soup.find_all(["code", "pre"])
            code_text   = "\n".join(b.get_text() for b in code_blocks)

            # Extract main content
            main = soup.find("main") or soup.find("article") or soup.find("body")
            body_text = main.get_text(separator="\n") if main else ""

            # Combine code + body
            combined = code_text + "\n\n" + body_text
            if combined.strip():
                yield combined

        except Exception as e:
            print(f"  Warning: Failed to scrape {url}: {e}")
```

class AfriCodeSource:
“”“Auto-scrape the pre-defined list of African API documentation URLs.”””

```
def __init__(self):
    try:
        import requests
        from bs4 import BeautifulSoup
        self.requests      = requests
        self.BeautifulSoup = BeautifulSoup
    except ImportError:
        raise ImportError("Run: pip install requests beautifulsoup4")

    self.urls = AFRICODE_URLS
    print(f"AfriCode source: scraping {len(self.urls)} African API documentation URLs")

def iter_documents(self) -> Iterator[str]:
    for url in tqdm(self.urls, desc="Scraping African APIs"):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (AfriCodeBot/1.0)"}
            resp    = self.requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            soup = self.BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            code_blocks = soup.find_all(["code", "pre"])
            code_text   = "\n".join(b.get_text() for b in code_blocks)

            main      = soup.find("main") or soup.find("article") or soup.find("body")
            body_text = main.get_text(separator="\n") if main else ""

            yield f"# Source: {url}\n\n{code_text}\n\n{body_text}"

        except Exception as e:
            print(f"  Warning: {url}: {e}")
```

class HuggingFaceSource:
“”“Load a HuggingFace dataset. Defaults to code datasets.”””

```
def __init__(self, dataset_name: str, config: str = None):
    try:
        from datasets import load_dataset
        self.load_dataset = load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"Loading HuggingFace dataset: {dataset_name} | config: {config}")
    self.dataset      = load_dataset(dataset_name, config, trust_remote_code=True)
    self.dataset_name = dataset_name

def iter_documents(self) -> Iterator[str]:
    split = "train" if "train" in self.dataset else list(self.dataset.keys())[0]
    data  = self.dataset[split]

    # Code datasets use "content", others use "text"
    text_keys = ["content", "text", "code", "body", "document"]

    for item in tqdm(data, desc=f"Loading {self.dataset_name}"):
        for key in text_keys:
            if key in item and item[key]:
                yield str(item[key])
                break
```

class QASource:
“””
Load Q&A pairs for fine-tuning.
Expects JSON files with format:
[
{“question”: “How do I integrate M-Pesa?”, “answer”: “Here is the code…”},
…
]
“””

```
def __init__(self, input_dir: str):
    self.input_dir = Path(input_dir)
    self.files     = list(self.input_dir.rglob("*.json"))
    if not self.files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")
    print(f"Found {len(self.files)} Q&A files")

def iter_documents(self) -> Iterator[str]:
    for filepath in tqdm(self.files, desc="Loading Q&A pairs"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    if "question" in item and "answer" in item:
                        # Format as instruction-following style
                        doc = (
                            f"### Question:\n{item['question']}\n\n"
                            f"### Answer:\n{item['answer']}\n"
                        )
                        yield doc
        except Exception as e:
            print(f"  Warning: {filepath}: {e}")
```

# —————————————————————————

# Dataset Builder — Tokenize & Save as Binary

# —————————————————————————

class DatasetBuilder:
“”“Tokenize documents and save to binary files for fast training.”””

```
def __init__(self, output_dir: str, val_frac: float = 0.05):
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.val_frac   = val_frac

    try:
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        raise ImportError("Run: pip install tiktoken")

def build(self, documents: List[str], output_name: str = "africode"):
    print(f"\nTokenizing {len(documents):,} documents...")

    all_tokens  = []
    total_chars = 0

    for doc in tqdm(documents, desc="Tokenizing"):
        tokens = self.enc.encode(doc, allowed_special={"<|endoftext|>"})
        tokens.append(self.enc.eot_token)  # document separator
        all_tokens.extend(tokens)
        total_chars += len(doc)

    all_tokens  = np.array(all_tokens, dtype=np.uint16)
    total_tokens = len(all_tokens)
    split_idx   = int(total_tokens * (1 - self.val_frac))

    train_tokens = all_tokens[:split_idx]
    val_tokens   = all_tokens[split_idx:]

    train_path = self.output_dir / f"{output_name}_train.bin"
    val_path   = self.output_dir / f"{output_name}_val.bin"
    meta_path  = self.output_dir / f"{output_name}_meta.json"

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    meta = {
        "dataset":         output_name,
        "total_tokens":    total_tokens,
        "train_tokens":    len(train_tokens),
        "val_tokens":      len(val_tokens),
        "total_chars":     total_chars,
        "total_documents": len(documents),
        "vocab_size":      self.enc.n_vocab,
        "encoding":        "gpt2",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  AfriCode Dataset Ready!")
    print(f"  Documents  : {len(documents):,}")
    print(f"  Characters : {total_chars:,}")
    print(f"  Tokens     : {total_tokens:,}")
    print(f"  Train      : {len(train_tokens):,} tokens → {train_path}")
    print(f"  Val        : {len(val_tokens):,} tokens  → {val_path}")
    print(f"{'='*55}\n")

    return str(train_path), str(val_path)
```

# —————————————————————————

# Dataset Analysis

# —————————————————————————

def analyze_dataset(documents: List[str]):
print(”\n— AfriCode Dataset Analysis —”)
lengths     = [len(d) for d in documents]
total_chars = sum(lengths)
total_words = sum(len(d.split()) for d in documents)
token_est   = int(total_words * 1.3)

```
print(f"  Documents     : {len(documents):,}")
print(f"  Total chars   : {total_chars:,}")
print(f"  Avg doc length: {total_chars // max(len(documents), 1):,} chars")
print(f"  Est. tokens   : ~{token_est:,}")

print(f"\n--- Training Readiness ---")
if token_est < 1_000_000:
    print(f"  ⚠️  Small ({token_est:,} tokens) — collect more data before serious training")
    print(f"  Tip: Add the-stack-smol Python dataset for instant volume boost")
elif token_est < 10_000_000:
    print(f"  ✓  Good ({token_est:,} tokens) — solid start for a 10–25M param model")
elif token_est < 100_000_000:
    print(f"  ✓✓ Large ({token_est:,} tokens) — ready for a serious 85M param model")
else:
    print(f"  🚀 Excellent ({token_est:,} tokens) — you're playing in the big leagues!")
print()
```

# —————————————————————————

# Main

# —————————————————————————

def main():
args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

```
cleaner = TextCleaner(min_length=args.min_length, max_length=args.max_length)
deduper = Deduplicator()

# Select source
if args.source == "files":
    source = FileSource(args.input_dir)
elif args.source == "web":
    source = WebSource(args.urls)
elif args.source == "hf":
    source = HuggingFaceSource(args.dataset, args.dataset_config)
elif args.source == "qa":
    source = QASource(args.input_dir)
elif args.source == "africode":
    source = AfriCodeSource()

# Process documents
documents = []
stats     = {"total": 0, "low_quality": 0, "duplicate": 0, "accepted": 0}

print("\nProcessing documents...")
for raw_doc in source.iter_documents():
    stats["total"] += 1
    cleaned = cleaner.process(raw_doc)

    if cleaned is None:
        stats["low_quality"] += 1
        continue

    if deduper.is_duplicate(cleaned):
        stats["duplicate"] += 1
        continue

    documents.append(cleaned)
    stats["accepted"] += 1

print(f"\n  Total processed : {stats['total']:,}")
print(f"  Low quality     : {stats['low_quality']:,}")
print(f"  Duplicates      : {stats['duplicate']:,}")
print(f"  Accepted        : {stats['accepted']:,}")

if not documents:
    print("\nERROR: No documents passed the filter.")
    print("Try lowering --min_length or check your data source.")
    sys.exit(1)

if args.analyze:
    analyze_dataset(documents)

# Tokenize and save
builder = DatasetBuilder(output_dir=args.output_dir, val_frac=args.val_frac)
train_path, val_path = builder.build(documents, output_name=args.output)

print(f"✓ Pipeline complete! Train your model with:")
print(f"  python train.py --data_train {train_path} --data_val {val_path}")
```

if **name** == “**main**”:
main()
