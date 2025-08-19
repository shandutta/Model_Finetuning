'''
python scripts/testing/compare_baseline_vs_lora.py --prompt "Which part of my source code includes the logic for how users like or pass on properties?" --ckpt outputs/qwen3b_lora_8bit/checkpoint-625
# or:
python scripts/testing/compare_baseline_vs_lora.py --prompt "Which part of my source code includes the logic for how users like or pass on properties?" --ckpt outputs/qwen3b_lora_8bit/adapter
# or:
python scripts/testing/compare_baseline_vs_lora.py --prompt "Which part of my source code includes the logic for how users like or pass on properties?"
'''

# compare_baseline_vs_lora.py
import os, glob, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModelForCausalLM  # ensure this is the class you use

MID = "Qwen/Qwen2.5-Coder-3B-Instruct"
OUT_DIR = "outputs/qwen3b_lora_8bit"

SYSTEM_STEER = (
    "You are a precise coding assistant. "
    "You cannot call tools or browse. "
    "Answer in plain text. Provide exact file paths, function names, and code snippets where possible. "
    "Give a complete, self-contained answer."
)

TOOLY_STRINGS = ["<|tool|>", "<|assistant|tool_call|>", "<tool_call>"]

def find_latest_checkpoint(root=OUT_DIR):
    cks = sorted(glob.glob(os.path.join(root, "checkpoint-*")),
                 key=lambda p: int(p.split("-")[-1]))  # assumes numeric suffix
    return cks[-1] if cks else None

def build_chat(tok, user_text, system_text=SYSTEM_STEER):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_baseline(quant="8bit"):
    if quant == "8bit":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    tok = AutoTokenizer.from_pretrained(MID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # temporary until we set eos_id below
    model = AutoModelForCausalLM.from_pretrained(
        MID, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    return tok, model

def load_lora_from_checkpoint(base_model, ckpt_dir):
    tuned = PeftModelForCausalLM.from_pretrained(base_model, ckpt_dir, device_map="auto")
    tuned.eval()
    return tuned

def extract_assistant_turn(text: str) -> str:
    # Return only what the assistant actually wrote after the last assistant tag.
    # Qwen chat format uses <|im_start|>assistant ... <|im_end|>
    start_tag = "<|im_start|>assistant"
    end_tag = "<|im_end|>"
    if start_tag in text:
        seg = text.split(start_tag)[-1]
        if end_tag in seg:
            seg = seg.split(end_tag)[0]
        # drop any leading newline/role markers
        return seg.lstrip()
    return text

def strip_tooly(text: str) -> str:
    for s in TOOLY_STRINGS:
        if s in text:
            text = text.split(s)[0]
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="User question in chat style")
    ap.add_argument("--ckpt", default=None, help="Path to LoRA checkpoint dir or adapter dir")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--min_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--quant", choices=["8bit","4bit"], default="8bit")
    args = ap.parse_args()

    # tokenizer + baseline
    tok, base_model = load_baseline(args.quant)

    # correct eos/pad ids for Qwen chat
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id or eos_id

    # resolve adapter/checkpoint dir
    ckpt_dir = args.ckpt or find_latest_checkpoint() or os.path.join(OUT_DIR, "adapter")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoint/adapter dir found at: {ckpt_dir}")

    # chat-formatted input
    chat_text = build_chat(tok, args.prompt)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=(args.temperature > 0.0),
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        repetition_penalty=1.05,
    )

    # ----- BASELINE -----
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=tok, device_map="auto")
    base_raw = base_pipe(chat_text, **gen_kwargs)[0]["generated_text"]
    base_out = extract_assistant_turn(strip_tooly(base_raw))

    # ----- LoRA-TUNED -----
    # fresh base for wrapping (avoid adapter state bleeding)
    _, base_for_lora = load_baseline(args.quant)
    lora_model = load_lora_from_checkpoint(base_for_lora, ckpt_dir)
    lora_pipe = pipeline("text-generation", model=lora_model, tokenizer=tok, device_map="auto")
    lora_raw = lora_pipe(chat_text, **gen_kwargs)[0]["generated_text"]
    lora_out = extract_assistant_turn(strip_tooly(lora_raw))

    # Pretty print
    sep = "\n" + "="*80 + "\n"
    print(sep + "USER PROMPT:\n" + args.prompt)
    print(sep + "[BASELINE]\n" + base_out.strip())
    print(sep + f"[LoRA TUNED @ {ckpt_dir}]\n" + lora_out.strip())
    print(sep)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
