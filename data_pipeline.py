import os
import re
import sys
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

AFRICODE_URLS = [
    "https://developer.safaricom.co.ke/APIs",
    "https://paystack.com/docs/api/",
    "https://paystack.com/docs/payments/accept-payments/",
    "https://developer.flutterwave.com/docs",
    "https://momodeveloper.mtn.com/docs",
    "https://engineering.paystack.com",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Zara Data Pipeline")
    parser.add_argument("--source", type=str, required=True,
                        choices=["files", "web", "hf", "qa", "africode"])
    parser.add_argument("--input_dir", type=str, default="./raw_data")
    parser.add_argument("--urls", type=str, default="africode_urls.txt")
    parser.add_argument("--dataset", type=str, default="bigcode/the-stack-smol")
    parser.add_argument("--dataset_config", type=str, default="python")
    parser.add_argument("--output", type=str, default="zara")
    parser.add_argument("--output_dir", type=str, default="./processed")
    parser.add_argument("--min_length", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=50000)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--analyze", action="store_true")
    return parser.parse_args()


class TextCleaner:
    def __init__(self, min_length=50, max_length=50000):
        self.min_length = min_length
        self.max_length = max_length

    def clean(self, text):
        if not text or not isinstance(text, str):
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text.strip()

    def is_valid(self, text):
        if len(text) < self.min_length:
            return False
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        return alpha_ratio >= 0.1

    def truncate(self, text):
        if len(text) > self.max_length:
            return text[: self.max_length]
        return text

    def process(self, text):
        text = self.clean(text)
        if not self.is_valid(text):
            return None
        return self.truncate(text)


class Deduplicator:
    def __init__(self):
        self.seen = set()

    def is_duplicate(self, text):
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in self.seen:
            return True
        self.seen.add(h)
        return False


class FileSource:
    EXTENSIONS = [".txt", ".md", ".py", ".js", ".php", ".java", ".json", ".ts"]

    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)
        self.files = []
        for ext in self.EXTENSIONS:
            self.files.extend(self.input_dir.rglob("*" + ext))
        if not self.files:
            raise FileNotFoundError("No files found in " + str(input_dir))
        print("Found " + str(len(self.files)) + " files")

    def iter_documents(self):
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
                                break
                else:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        yield f.read()
            except Exception as e:
                print("Warning: could not read " + str(filepath) + ": " + str(e))


class WebSource:
    def __init__(self, urls_file):
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Run: pip install requests beautifulsoup4")
        self.requests = requests
        self.BeautifulSoup = BeautifulSoup

        with open(urls_file, "r", encoding="utf-8") as f:
            self.urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        print("Loaded " + str(len(self.urls)) + " URLs")

    def iter_documents(self):
        for url in tqdm(self.urls, desc="Scraping URLs"):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = self.requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                soup = self.BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                code_blocks = soup.find_all(["code", "pre"])
                code_text = "\n".join(b.get_text() for b in code_blocks)
                main = soup.find("main") or soup.find("article") or soup.find("body")
                body_text = main.get_text(separator="\n") if main else ""
                combined = code_text + "\n\n" + body_text
                if combined.strip():
                    yield combined
            except Exception as e:
                print("Warning: failed to scrape " + url + ": " + str(e))


class AfriCodeSource:
    def __init__(self):
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Run: pip install requests beautifulsoup4")
        self.requests = requests
        self.BeautifulSoup = BeautifulSoup
        print("Scraping " + str(len(AFRICODE_URLS)) + " African API documentation URLs")

    def iter_documents(self):
        for url in tqdm(AFRICODE_URLS, desc="Scraping African APIs"):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = self.requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                soup = self.BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                code_blocks = soup.find_all(["code", "pre"])
                code_text = "\n".join(b.get_text() for b in code_blocks)
                main = soup.find("main") or soup.find("article") or soup.find("body")
                body_text = main.get_text(separator="\n") if main else ""
                yield "# Source: " + url + "\n\n" + code_text + "\n\n" + body_text
            except Exception as e:
                print("Warning: " + url + ": " + str(e))


class HuggingFaceSource:
    def __init__(self, dataset_name, config=None):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Run: pip install datasets")
        self.load_dataset = load_dataset
        print("Loading dataset: " + dataset_name + " config: " + str(config))
        self.dataset = load_dataset(dataset_name, config, trust_remote_code=True)
        self.dataset_name = dataset_name

    def iter_documents(self):
        split = "train" if "train" in self.dataset else list(self.dataset.keys())[0]
        data = self.dataset[split]
        text_keys = ["content", "text", "code", "body", "document"]
        for item in tqdm(data, desc="Loading " + self.dataset_name):
            for key in text_keys:
                if key in item and item[key]:
                    yield str(item[key])
                    break


class QASource:
    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)
        self.files = list(self.input_dir.rglob("*.json"))
        if not self.files:
            raise FileNotFoundError("No JSON files found in " + str(input_dir))
        print("Found " + str(len(self.files)) + " Q&A files")

    def iter_documents(self):
        for filepath in tqdm(self.files, desc="Loading Q&A pairs"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            yield "### Question:\n" + item["question"] + "\n\n### Answer:\n" + item["answer"] + "\n"
            except Exception as e:
                print("Warning: " + str(filepath) + ": " + str(e))


class DatasetBuilder:
    def __init__(self, output_dir, val_frac=0.05):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_frac = val_frac
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
        except ImportError:
            raise ImportError("Run: pip install tiktoken")

    def build(self, documents, output_name="zara"):
        print("Tokenizing " + str(len(documents)) + " documents...")
        all_tokens = []
        total_chars = 0
        for doc in tqdm(documents, desc="Tokenizing"):
            tokens = self.enc.encode(doc, allowed_special={"<|endoftext|>"})
            if hasattr(self.enc, "eot_token"):
                tokens.append(self.enc.eot_token)
            all_tokens.extend(tokens)
            total_chars += len(doc)

        all_tokens = np.array(all_tokens, dtype=np.uint16)
        total_tokens = len(all_tokens)
        split_idx = int(total_tokens * (1 - self.val_frac))

        train_tokens = all_tokens[:split_idx]
        val_tokens = all_tokens[split_idx:]

        train_path = self.output_dir / (output_name + "_train.bin")
        val_path = self.output_dir / (output_name + "_val.bin")
        meta_path = self.output_dir / (output_name + "_meta.json")

        train_tokens.tofile(train_path)
        val_tokens.tofile(val_path)

        meta = {
            "dataset": output_name,
            "total_tokens": total_tokens,
            "train_tokens": len(train_tokens),
            "val_tokens": len(val_tokens),
            "total_chars": total_chars,
            "total_documents": len(documents),
            "vocab_size": self.enc.n_vocab,
            "encoding": "gpt2",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print("Dataset ready!")
        print("  Documents : " + str(len(documents)))
        print("  Tokens    : " + str(total_tokens))
        print("  Train     : " + str(len(train_tokens)) + " -> " + str(train_path))
        print("  Val       : " + str(len(val_tokens)) + " -> " + str(val_path))

        return str(train_path), str(val_path)


def analyze_dataset(documents):
    print("\n--- Dataset Analysis ---")
    lengths = [len(d) for d in documents]
    total_chars = sum(lengths)
    total_words = sum(len(d.split()) for d in documents)
    token_est = int(total_words * 1.3)

    print("  Documents     : " + str(len(documents)))
    print("  Total chars   : " + str(total_chars))
    print("  Avg length    : " + str(total_chars // max(len(documents), 1)) + " chars")
    print("  Est. tokens   : " + str(token_est))

    if token_est < 1000000:
        print("  Status: Small dataset - collect more data")
    elif token_est < 10000000:
        print("  Status: Good - solid for a 10-25M param model")
    elif token_est < 100000000:
        print("  Status: Large - ready for an 85M param model")
    else:
        print("  Status: Excellent - serious training scale")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cleaner = TextCleaner(min_length=args.min_length, max_length=args.max_length)
    deduper = Deduplicator()

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
    else:
        raise ValueError("Unsupported source: " + args.source)

    documents = []
    stats = {"total": 0, "low_quality": 0, "duplicate": 0, "accepted": 0}

    print("Processing documents...")
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

    print("Total     : " + str(stats["total"]))
    print("Rejected  : " + str(stats["low_quality"]))
    print("Duplicate : " + str(stats["duplicate"]))
    print("Accepted  : " + str(stats["accepted"]))

    if not documents:
        print("ERROR: No documents passed the filter.")
        sys.exit(1)

    if args.analyze:
        analyze_dataset(documents)

    builder = DatasetBuilder(output_dir=args.output_dir, val_frac=args.val_frac)
    train_path, val_path = builder.build(documents, output_name=args.output)

    print("Pipeline complete!")
    print("Train with:")
    print("  python train.py --data_train " + train_path + " --data_val " + val_path)


if __name__ == "__main__":
    main()
