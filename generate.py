import argparse
import torch
from model import ModelConfig, TransformerLM

# Legacy code-completion templates from Zara's earlier "AfriCode" direction.
# These are raw code continuations, NOT the "### Question:\n...\n\n### Answer:\n"
# format Zara is actually trained on now (see Tools/ and data_pipeline.py's
# QASource). They're kept for reference/compatibility, but if your current
# checkpoint was trained on the cybersecurity Q&A corpus, results from these
# will likely be weaker than CYBERSECURITY_TEMPLATES below, which are phrased
# to match the actual training format.
CODE_TEMPLATES = {
    "mpesa": "# M-Pesa STK Push integration in Python\nimport requests\n\ndef stk_push_request():\n    pass\n",
    "paystack": "# Paystack payment integration\nimport requests\n\nPAYSTACK_SECRET = ",
    "flutterwave": "# Flutterwave payment integration\nimport requests\n\ndef initiate_payment(",
    "mtn_momo": "# MTN Mobile Money API integration\nimport requests\n\ndef mtn_payment(",
    "ussd": "# USSD menu handler\n\ndef handle_ussd_request(session_id, phone_number, text):\n",
    "airtel": "# Airtel Money integration\nimport requests\n\ndef airtel_payment(",
}

# Cybersecurity Q&A templates -- these match the "### Question:\n...\n\n### Answer:\n"
# format used by data_pipeline.py's QASource and by zara_agent.py's _generate(),
# so they exercise the model the way it's actually meant to be used.
CYBERSECURITY_TEMPLATES = {
    "sim_swap": "What is SIM swap fraud and how can African mobile money providers detect it?",
    "mpesa_fraud": "What are common M-Pesa fraud patterns and how should they be detected?",
    "ransomware": "What are the immediate steps to take after a ransomware attack hits a bank in Kenya?",
    "ndpr": "What are the key requirements of Nigeria's NDPR for a fintech company?",
    "popia": "What does POPIA require for a company operating in South Africa?",
    "phishing": "How does phishing targeting African mobile money users typically work?",
    "incident_response": "What are the first three steps after confirming a data breach?",
}

TEMPLATES = {**CODE_TEMPLATES, **CYBERSECURITY_TEMPLATES}


def parse_args():
    parser = argparse.ArgumentParser(description="Zara by Nexara - Cybersecurity Q&A / Code Generation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--template", type=str, default=None, choices=list(TEMPLATES.keys()))
    parser.add_argument("--qa_format", action="store_true",
                         help="Wrap --prompt in the '### Question:\\n...\\n\\n### Answer:\\n' format "
                              "used during training. Automatically applied for CYBERSECURITY_TEMPLATES; "
                              "use this flag to also apply it to a free-form --prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ModelConfig(**ckpt["cfg"])
    model = TransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print("Zara loaded | " + str(model.num_parameters()) + " params | step " + str(step))
    return model, cfg


def generate_code(model, enc, prompt, device, max_new_tokens=300, temperature=0.7, top_k=40, qa_format=False):
    if qa_format:
        prompt = "### Question:\n" + prompt.strip() + "\n\n### Answer:\n"
    tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        eos_token_id=getattr(enc, "eot_token", None),
    )
    decoded = enc.decode(out[0].tolist())
    if qa_format:
        # Show only the generated continuation, and stop at a hallucinated new turn.
        answer = decoded[len(prompt):]
        if "### Question:" in answer:
            answer = answer.split("### Question:")[0]
        return answer.strip()
    return decoded


def interactive_mode(model, enc, device, args):
    print("\nZara by Nexara - Interactive Mode")
    print("Type a template name (" + ", ".join(TEMPLATES.keys()) + "),")
    print("or type any cybersecurity question directly (it will be sent in Q&A format).")
    print("Type 'quit' to exit")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input.lower() in TEMPLATES:
            prompt = TEMPLATES[user_input.lower()]
            qa_format = user_input.lower() in CYBERSECURITY_TEMPLATES
            print("Using template: " + user_input)
        else:
            prompt = user_input
            qa_format = True  # free-form input during interactive mode is almost always a question

        print("\n" + "=" * 50)
        result = generate_code(
            model, enc, prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            qa_format=qa_format,
        )
        print(result)
        print("=" * 50)


def main():
    args = parse_args()

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        raise ImportError("Run: pip install tiktoken")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Device: " + device)

    model, cfg = load_model(args.checkpoint, device)

    if args.interactive:
        interactive_mode(model, enc, device, args)
        return

    if args.template:
        prompt = TEMPLATES[args.template]
        qa_format = args.template in CYBERSECURITY_TEMPLATES
        print("Template: " + args.template)
    elif args.prompt:
        prompt = args.prompt
        qa_format = args.qa_format
    else:
        print("No prompt given. Running all cybersecurity Q&A templates...\n")
        for name, template_prompt in CYBERSECURITY_TEMPLATES.items():
            print("=" * 55)
            print("Template: " + name.upper())
            print("=" * 55)
            result = generate_code(
                model, enc, template_prompt, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                qa_format=True,
            )
            print(result)
        return

    print("Prompt: " + prompt)
    print("=" * 55)
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print("\n--- Sample " + str(i + 1) + "/" + str(args.num_samples) + " ---")
        result = generate_code(
            model, enc, prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            qa_format=qa_format,
        )
        print(result)


if __name__ == "__main__":
    main()