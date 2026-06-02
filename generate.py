import argparse
import torch
from model import ModelConfig, TransformerLM

TEMPLATES = {
    "mpesa": "# M-Pesa STK Push integration in Python\nimport requests\n\ndef stk_push_request():\n    pass\n",
    "paystack": "# Paystack payment integration\nimport requests\n\nPAYSTACK_SECRET = ",
    "flutterwave": "# Flutterwave payment integration\nimport requests\n\ndef initiate_payment(",
    "mtn_momo": "# MTN Mobile Money API integration\nimport requests\n\ndef mtn_payment(",
    "ussd": "# USSD menu handler\n\ndef handle_ussd_request(session_id, phone_number, text):\n",
    "airtel": "# Airtel Money integration\nimport requests\n\ndef airtel_payment(",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Zara by Nexara - Code Generation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--template", type=str, default=None, choices=list(TEMPLATES.keys()))
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


def generate_code(model, enc, prompt, device, max_new_tokens=300, temperature=0.7, top_k=40):
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(out[0].tolist())


def interactive_mode(model, enc, device, args):
    print("\nZara by Nexara - Interactive Mode")
    print("Type a prompt or a template name: " + ", ".join(TEMPLATES.keys()))
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
            print("Using template: " + user_input)
        else:
            prompt = user_input

        print("\n" + "=" * 50)
        result = generate_code(
            model, enc, prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
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
        print("Template: " + args.template)
    elif args.prompt:
        prompt = args.prompt
    else:
        print("No prompt given. Running all templates...\n")
        for name, template_prompt in TEMPLATES.items():
            print("=" * 55)
            print("Template: " + name.upper())
            print("=" * 55)
            result = generate_code(
                model, enc, template_prompt, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
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
        )
        print(result)


if __name__ == "__main__":
    main()
