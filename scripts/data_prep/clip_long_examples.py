# scripts/data_prep/clip_long_examples.py
import json, sys
from transformers import AutoTokenizer
MID = "Qwen/Qwen2.5-Coder-3B-Instruct"
tok = AutoTokenizer.from_pretrained(MID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

inp, outp, max_len = sys.argv[1], sys.argv[2], int(sys.argv[3])  # e.g., 2048
dropped = kept = 0

with open(inp, "r", encoding="utf-8", errors="ignore") as fin, open(outp, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip(): continue
        ex = json.loads(line)
        text = ex.get("text")
        if text is None and "messages" in ex:
            # if you’re storing chat messages and formatting in-trainer
            # you can pre-template here to measure length more accurately:
            # text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            # But if you already have ex["text"], just use it:
            pass
        if text is None: continue
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) <= max_len:
            fout.write(line); kept += 1
        else:
            # Option A (drop): skip it
            # dropped += 1; continue

            # Option B (truncate): keep head up to max_len
            ex["text"] = tok.decode(ids[:max_len], skip_special_tokens=False)
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n"); kept += 1
            dropped += 1

print(f"Kept: {kept}  |  Truncated: {dropped}  |  Max len: {max_len}")
