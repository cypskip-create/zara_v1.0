# Zara 🛡️

**Africa's Cybersecurity Brain**

Zara is a GPT-style language model paired with a rule-based security tool
suite, purpose-built for the African cybersecurity, fraud, and compliance
landscape. It is not a general-purpose assistant fine-tuned with a few
African examples bolted on — it's trained and structured around the threat
patterns, regulatory frameworks, and financial infrastructure specific to
the continent.

-----

## Why Zara?

Most cybersecurity tooling and AI assistants are built around Western
threat models, Western compliance frameworks (GDPR, HIPAA, SOC 2), and
Western financial rails (card networks, ACH, wire transfer). African
organizations face a different, under-served threat landscape:

- **SIM swap fraud** — the single most damaging attack against mobile
  money users, because SMS-based OTP is still the dominant second factor
- **Mobile money fraud** — M-Pesa, MTN MoMo, Airtel Money-specific attack
  patterns (wrong-number reversal scams, agent collusion, float draining)
- **Regional compliance** — Nigeria's NDPR/NDPA, Kenya's Data Protection
  Act, South Africa's POPIA, Ghana's Data Protection Act, plus financial
  sector frameworks from the CBN, CBK, and Bank of Ghana
- **Agent banking fraud** — KYC fraud, float reconciliation gaps, and
  terminal-level attacks specific to agent-assisted banking models
- **Under-resourced security teams** — most African SMEs and even mid-sized
  fintechs don't have a dedicated SOC, so tooling needs to be actionable
  without assuming a large security headcount

No major AI lab is focused on this. Zara fills that gap.

-----

## Project Structure

```
zara_v1.0/
├── model.py                 # Transformer architecture (TransformerLM / ModelConfig)
├── train.py                  # Training loop, checkpointing, sampling
├── generate.py                 # Standalone text generation (Q&A + legacy code templates)
├── data_pipeline.py             # Data collection, cleaning, and token-budgeted mixing
├── zara_agent.py                  # The agent: routes user input to tools or the LM
├── africode_urls.txt               # Source URLs for --source web scraping
├── Tools/                            # Rule-based security tools Zara can call
│   ├── base_tool.py                     # Base class every tool inherits from
│   ├── tool_registry.py                 # Central registry + keyword-based intent routing
│   ├── threat_detector.py               # Text/log threat pattern matching
│   ├── fraud_detector.py                # Transaction risk scoring
│   ├── security_auditor.py              # Static code vulnerability scanning
│   ├── incident_responder.py            # Country-specific incident response playbooks
│   ├── compliance_checker.py            # NDPR/NDPA, Kenya DPA, POPIA, Ghana DPA, CBN/CBK/BoG
│   └── vulnerability_scanner.py         # System security posture checklist scoring
├── processed/                              # Tokenized training data (generated)
└── checkpoints/                              # Saved model weights (generated)
```

-----

## Two ways Zara answers a question

Zara is a hybrid system, not a pure language model wrapper:

1. **Tool path (deterministic).** `ToolRegistry.detect_intent()` matches the
   user's input against keyword patterns for each of the six tools above.
   If a tool matches, it runs and returns a structured, deterministic
   result (risk score, compliance gaps, playbook steps, etc.), which
   `zara_agent.py` formats into readable text. This path never depends on
   model quality — a freshly initialized model gives the exact same tool
   output as a fully trained one.
2. **Language model path (learned).** If no tool matches, `zara_agent.py`
   asks Zara's trained language model directly, using the
   `### Question:\n...\n\n### Answer:\n` format it was trained on. If the
   checkpoint isn't loaded, or the generation is empty/degenerate (a
   repetition-collapse heuristic catches obvious small-model failure
   modes), Zara falls back to a small set of curated responses rather than
   showing broken output.

This means Zara is useful for the deterministic tool cases even before the
language model is fully trained, and gets progressively better at
open-ended questions as training rounds improve the underlying model.

-----

## Quick Start

### 1. Install Dependencies

```bash
pip install torch tiktoken numpy datasets requests beautifulsoup4 tqdm
```

### 2. Build a training dataset

**Single source (e.g. a hand-written cybersecurity Q&A set):**

```bash
python data_pipeline.py \
  --source qa \
  --input_dir ./qa_pairs \
  --output zara_qa \
  --output_dir ./processed \
  --repeat 4 \
  --analyze
```

`--repeat` matters here: a small, high-value QA corpus (tens to low
hundreds of pairs) needs repetition to actually be learned by a small
model, rather than being statistically invisible next to other data.

**Multi-source, token-budgeted mix (recommended for anything beyond a
single QA file):**

```bash
python data_pipeline.py \
  --mix_spec ./mix_spec.json \
  --output zara_mix \
  --output_dir ./processed \
  --analyze
```

Where `mix_spec.json` looks like:

```json
[
  {"source": "qa", "input_dir": "./qa_pairs/cybersecurity", "repeat": 4},
  {"source": "qa", "input_dir": "./qa_pairs/smalltalk", "repeat": 8},
  {"source": "hf", "dataset": "wikitext", "dataset_config": "wikitext-103-raw-v1",
   "token_ratio": 0.3, "doc_pool": 20000}
]
```

`token_ratio` controls the mix by **token count**, not document count —
this is deliberate. A fixed document count silently lets a long-form
source (like WikiText, where each document is a full article) drown out a
short-form source (like QA pairs) even at document counts that look
reasonable on paper. `repeat`-based entries are always fully included;
`token_ratio`-based entries are trimmed to hit a budget computed relative
to the repeat-based entries' total token count. All documents are shuffled
before tokenizing, regardless of path, which breaks up long topic-
contiguous runs that otherwise contribute to memorization and looping in a
small model.

### 3. Train

```bash
python train.py \
  --data_train ./processed/zara_mix_train.bin \
  --data_val ./processed/zara_mix_val.bin \
  --max_steps 15000 \
  --run_name zara_r5
```

`--run_name` controls the final checkpoint filename (`<run_name>.pt`). If
omitted, it defaults to `zara_step<N>.pt` using the actual final step
count — either way, every round gets its own name instead of every run
silently overwriting the same file.

To resume from an existing checkpoint:

```bash
python train.py \
  --data_train ./processed/zara_mix_train.bin \
  --data_val ./processed/zara_mix_val.bin \
  --resume ./checkpoints/best_model.pt \
  --out_dir ./checkpoints \
  --max_steps 30000 \
  --run_name zara_r6
```

Checkpoints save every `--save_every` steps (default 500) with only the
last 2 kept automatically (`cleanup_old_checkpoints`), plus a
`best_model.pt` that's overwritten whenever validation loss improves. If
`--out_dir` points directly at a mounted Google Drive path, this all
happens on Drive with no separate copy step.

### 4. Generate / test

```bash
# Cybersecurity Q&A templates (matches the actual training format)
python generate.py --checkpoint checkpoints/best_model.pt --template sim_swap
python generate.py --checkpoint checkpoints/best_model.pt --template ransomware

# Free-form question
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "How do I detect agent banking fraud?" \
  --qa_format

# Interactive mode
python generate.py --checkpoint checkpoints/best_model.pt --interactive
```

### 5. Use the full agent (tools + language model)

```python
from zara_agent import ZaraAgent

zara = ZaraAgent(checkpoint_path="checkpoints/best_model.pt")

# Routes to compliance_checker tool (deterministic)
print(zara.chat("What are Kenya's data protection requirements for a fintech?"))

# Routes to incident_responder tool (deterministic)
print(zara.chat("We just got hit with ransomware in Nigeria, what do we do?"))

# No tool match -> falls through to the language model
print(zara.chat("Why is SMS OTP risky in the African mobile money context?"))
```

-----

## The Tools

| Tool | What it does |
|---|---|
| `threat_detector` | Keyword-pattern matching against text/logs for SIM swap, phishing, credential stuffing, social engineering, malware indicators, data exfiltration, ransomware, and mobile money fraud patterns |
| `fraud_detector` | Weighted rule-based risk scoring for a transaction (SIM recently changed, new device, unusual hour, rapid transactions, round amounts, etc.) with a risk level and recommended action |
| `security_auditor` | Regex-based static scan of source code for hardcoded secrets, SQL injection risk, exposed M-Pesa/Paystack credentials, missing rate limiting, and other common vulnerability classes |
| `incident_responder` | Structured playbooks (immediate/short-term/recovery steps) for ransomware, data breach, SIM swap, phishing, and account takeover, with country-specific regulatory reporting requirements for Nigeria, Kenya, South Africa, and Ghana |
| `compliance_checker` | Checks a set of completed controls against NDPR/NDPA (Nigeria), Kenya DPA, POPIA (South Africa), Ghana DPA, and financial-sector frameworks (CBN, CBK, Bank of Ghana), returning a compliance score and gap list |
| `vulnerability_scanner` | Checklist-style system security posture scoring (MFA, network segmentation, backups, encryption, logging, etc.) with a letter grade |

Each tool inherits from `Tools/base_tool.py`'s `BaseTool`, which wraps
`run()` with timing and error handling via `execute()`. `ToolRegistry`
(`Tools/tool_registry.py`) is the central place new tools get registered
and where keyword-based intent detection lives.

-----

## Model Sizes

| Config | Params (approx, weight-tied) | Good for | GPU needed |
|---|---|---|---|
| Small (default: d_model=512, 8 layers, 8 heads) | ~51M | Colab free tier, iterative rounds | T4 (free) |
| Medium (d_model=768, 12 layers, 12 heads) | ~85M | Serious training | A100 |
| Large (d_model=1024+, 16+ layers) | ~300M+ | Production quality | A100 x2+ |

Note on parameter counts: `model.num_parameters()` counts unique
parameters, and `lm_head` is weight-tied to the token embedding table, so
the reported count is lower than a naive layer-by-layer sum would suggest
— this is expected, not a bug.

### Scale up:

```bash
python train.py \
  --data_train ./processed/zara_mix_train.bin \
  --data_val ./processed/zara_mix_val.bin \
  --d_model 768 --n_heads 12 --n_layers 12 --d_ff 3072 \
  --context_len 1024 --batch_size 8 --max_steps 50000
```

-----

## Training on Google Colab (Free Tier)

```python
# Runtime > Change runtime type > T4 GPU
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/cypskip-create/zara_v1.0.git
%cd zara_v1.0
!pip install -q torch tiktoken numpy datasets requests beautifulsoup4 tqdm

# Build the dataset (see mix_spec.json example above)
!python data_pipeline.py \
    --mix_spec ./mix_spec.json \
    --output zara_mix \
    --output_dir ./processed \
    --analyze

# Train, saving checkpoints straight to Drive
!python train.py \
    --data_train ./processed/zara_mix_train.bin \
    --data_val ./processed/zara_mix_val.bin \
    --out_dir "/content/drive/MyDrive/Nexara/zara_checkpoints" \
    --max_steps 15000 \
    --run_name zara_r5

# Test it
!python generate.py \
    --checkpoint "/content/drive/MyDrive/Nexara/zara_checkpoints/best_model.pt" \
    --template sim_swap
```

Free Colab sessions disconnect unpredictably — pointing `--out_dir`
directly at a mounted Drive path means checkpoints (and the automatic
2-most-recent cleanup) survive a disconnect with no extra steps.

-----

## Roadmap

- [x] Base transformer architecture
- [x] Rule-based security tool suite (6 tools, African-context-specific)
- [x] Token-budgeted multi-source data mixing
- [x] Language model actually wired into the agent's chat path
- [ ] Model-driven tool routing (structured `<think>` traces + JSON tool
      calls generated by the model itself, with fallback to the current
      keyword-based routing)
- [ ] REST API wrapper (Flask)
- [ ] Beta launch
- [ ] Expand `fraud_detector.py`'s rule coverage as new transaction fields
      are identified
- [ ] Web interface

-----

## Supported Countries (compliance & incident response)

| Country | Data Protection | Financial Sector Cybersecurity |
|---|---|---|
| Nigeria | NDPR / NDPA (NITDA / NDPC) | CBN Cybersecurity Framework |
| Kenya | Data Protection Act 2019 (ODPC) | CBK Cybersecurity Guidelines |
| South Africa | POPIA (Information Regulator) | — |
| Ghana | Data Protection Act 2012 (Data Protection Commission) | Bank of Ghana Cyber & Information Security Directive |

-----

*Built for African organizations, by Nexara.*