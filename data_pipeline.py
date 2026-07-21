import os
import re
import sys
import json
import random
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Shuffle seed. Documents are always shuffled before tokenizing -- this is what "
                             "prevents a model from seeing long unbroken topic-contiguous runs of a single "
                             "source, which is a real cause of memorization/looping in small models.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat the collected document set this many times before shuffling. Useful when "
                             "--source is a small, high-value corpus (e.g. a hand-written QA set) that would "
                             "otherwise be statistically invisible if later mixed with a much larger corpus.")
    parser.add_argument("--mix_spec", type=str, default=None,
                        help="Path to a JSON file describing a MULTI-SOURCE, TOKEN-BUDGETED mix, e.g.:\n"
                             '[{"source": "qa", "input_dir": "./qa_data", "repeat": 4, "token_ratio": 0.5},\n'
                             ' {"source": "qa", "input_dir": "./smalltalk_data", "repeat": 8, "token_ratio": 0.1},\n'
                             ' {"source": "hf", "dataset": "wikitext", "dataset_config": "wikitext-103-raw-v1", '
                             '"token_ratio": 0.4, "doc_pool": 20000}]\n'
                             "token_ratio values should sum to ~1.0 and control TOKEN counts, not document "
                             "counts -- this is deliberate, since a fixed document count silently lets a "
                             "long-form source (like WikiText) drown out a short-form one (like QA pairs) even "
                             "at document counts that look reasonable. When --mix_spec is set, --source is "
                             "ignored and this drives collection instead.")
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
    def __init__(self, dataset_name, config=None, max_docs=None):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Run: pip install datasets")
        self.load_dataset = load_dataset
        print("Loading dataset: " + dataset_name + " config: " + str(config))
        self.dataset = load_dataset(dataset_name, config, trust_remote_code=True)
        self.dataset_name = dataset_name
        self.max_docs = max_docs

    def iter_documents(self):
        split = "train" if "train" in self.dataset else list(self.dataset.keys())[0]
        data = self.dataset[split]
        text_keys = ["content", "text", "code", "body", "document"]
        count = 0
        for item in tqdm(data, desc="Loading " + self.dataset_name):
            if self.max_docs is not None and count >= self.max_docs:
                break
            for key in text_keys:
                if key in item and item[key]:
                    yield str(item[key])
                    count += 1
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


def build_source(source_type, **kwargs):
    if source_type == "files":
        return FileSource(kwargs.get("input_dir", "./raw_data"))
    elif source_type == "web":
        return WebSource(kwargs.get("urls", "africode_urls.txt"))
    elif source_type == "hf":
        return HuggingFaceSource(
            kwargs.get("dataset", "bigcode/the-stack-smol"),
            kwargs.get("dataset_config"),
            max_docs=kwargs.get("doc_pool"),
        )
    elif source_type == "qa":
        return QASource(kwargs.get("input_dir", "./raw_data"))
    elif source_type == "africode":
        return AfriCodeSource()
    else:
        raise ValueError("Unsupported source: " + source_type)


def collect_documents(source, cleaner, deduper):
    """Stream, clean, and dedup documents from a source. Returns (documents, stats)."""
    documents = []
    stats = {"total": 0, "low_quality": 0, "duplicate": 0, "accepted": 0}
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
    return documents, stats


def build_mix(mix_spec_path, cleaner):
    """
    Token-budgeted multi-source mixing. See --mix_spec help text for the
    expected JSON shape. Each spec entry is either:
      - "repeat": N       -> deterministic, fully included, repeated N times
      - "token_ratio": R  -> trimmed (at the document level, after shuffling
                              that source's own document pool) to hit a token
                              budget computed relative to the repeat-based
                              sources' total token count.
    Shuffling happens at the very end, across the ENTIRE combined document
    list, which is what actually breaks up topic-contiguous blocks and is
    the main lever against memorization/looping in a small model.
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError("Run: pip install tiktoken")
    enc = tiktoken.get_encoding("gpt2")

    with open(mix_spec_path, "r", encoding="utf-8") as f:
        specs = json.load(f)

    fixed_docs = []
    fixed_tokens = 0
    ratio_groups = []  # (spec, documents) pairs pending token-budget trim

    for spec in specs:
        deduper = Deduplicator()
        source = build_source(spec["source"], **spec)
        docs, stats = collect_documents(source, cleaner, deduper)
        print("[" + spec["source"] + "] total=" + str(stats["total"]) +
              " accepted=" + str(stats["accepted"]) +
              " duplicate=" + str(stats["duplicate"]) +
              " low_quality=" + str(stats["low_quality"]))

        if "repeat" in spec:
            repeated = docs * int(spec["repeat"])
            fixed_docs.extend(repeated)
            for d in repeated:
                fixed_tokens += len(enc.encode(d, allowed_special={"<|endoftext|>"})) + 1
        elif "token_ratio" in spec:
            random.shuffle(docs)  # sample varied documents, not just source order
            ratio_groups.append((spec, docs))
        else:
            # No mixing directive given -> treat as fully included, like "repeat": 1
            fixed_docs.extend(docs)
            for d in docs:
                fixed_tokens += len(enc.encode(d, allowed_special={"<|endoftext|>"})) + 1

    print("Fixed (repeat-based) token total: " + str(fixed_tokens))

    ratio_sum = sum(spec.get("token_ratio", 0) for spec, _ in ratio_groups)
    all_docs = list(fixed_docs)
    if ratio_groups and ratio_sum > 0:
        # grand_total * (1 - ratio_sum) = fixed_tokens  =>  grand_total = fixed_tokens / (1 - ratio_sum)
        denom = max(1e-6, (1 - ratio_sum)) if ratio_sum < 1 else 1e-6
        grand_total = fixed_tokens / denom
        for spec, docs in ratio_groups:
            budget = int(grand_total * spec["token_ratio"])
            used_tokens = 0
            used_docs = []
            for d in docs:
                if used_tokens >= budget:
                    break
                toks = len(enc.encode(d, allowed_special={"<|endoftext|>"})) + 1
                used_docs.append(d)
                used_tokens += toks
            print("[" + spec["source"] + "] token_ratio=" + str(spec["token_ratio"]) +
                  " -> used " + str(len(used_docs)) + " docs / " + str(used_tokens) +
                  " tokens (budget was " + str(budget) + ")")
            all_docs.extend(used_docs)
    elif ratio_groups:
        print("WARNING: token_ratio specs present but ratio_sum <= 0, skipping them entirely.")

    random.shuffle(all_docs)
    print("Final mixed document count: " + str(len(all_docs)))
    return all_docs


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
    random.seed(args.seed)

    cleaner = TextCleaner(min_length=args.min_length, max_length=args.max_length)

    if args.mix_spec:
        documents = build_mix(args.mix_spec, cleaner)
        if not documents:
            print("ERROR: No documents produced by --mix_spec.")
            sys.exit(1)
    else:
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

        print("Processing documents...")
        documents, stats = collect_documents(source, cleaner, deduper)

        print("Total     : " + str(stats["total"]))
        print("Rejected  : " + str(stats["low_quality"]))
        print("Duplicate : " + str(stats["duplicate"]))
        print("Accepted  : " + str(stats["accepted"]))

        if not documents:
            print("ERROR: No documents passed the filter.")
            sys.exit(1)

        if args.repeat > 1:
            documents = documents * args.repeat
            print("Repeated documents x" + str(args.repeat) + " -> " + str(len(documents)) + " total")

        # Shuffle before tokenizing. Without this, documents are concatenated in
        # their original source order, so a model training on the resulting
        # stream sees long unbroken topic-contiguous runs -- a real contributor
        # to memorization and looping in small models. This is the same fix
        # applied inside build_mix() for the multi-source case.
        random.shuffle(documents)

    if args.analyze:
        analyze_dataset(documents)

    builder = DatasetBuilder(output_dir=args.output_dir, val_frac=args.val_frac)
    train_path, val_path = builder.build(documents, output_name=args.output)

    print("Pipeline complete!")
    print("Train with:")
    print("  python train.py --data_train " + train_path + " --data_val " + val_path)


if __name__ == "__main__":
    main()