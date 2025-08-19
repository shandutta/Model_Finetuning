# scripts/data_prep/len_scan.py
import sys, json
from transformers import AutoTokenizer
from datasets import load_dataset

MID = "Qwen/Qwen2.5-Coder-3B-Instruct"
tok = AutoTokenizer.from_pretrained(MID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None: 
    tok.pad_token = tok.eos_token

path = sys.argv[1]
lens = []

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)
        # prefer "text", else build from "messages"
        if "text" in ex and ex["text"]:
            text = ex["text"]
        elif "messages" in ex and ex["messages"]:
            text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        else:
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        lens.append(len(ids))

lens.sort()
def pct(p): 
    i = int(p/100 * len(lens))
    return lens[min(len(lens)-1, i)]

print(f"count={len(lens)}  min={lens[0]}  p50={pct(50)}  p90={pct(90)}  p95={pct(95)}  p99={pct(99)}  max={lens[-1]}")

'''
ds = load_dataset("json", data_files="data/train.jsonl", split="train")
def fmt(e):
    if e.get("text"):                  # prefer prebuilt/clipped text
        return {"text": e["text"]}
    e["text"] = tok.apply_chat_template(
        e["messages"], tokenize=False, add_generation_prompt=False
    )
    return e
ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "text"])
assert all(len(t) > 0 for t in ds["text"]), "Found empty text rows in train set"

eval_ds = None
try:
    eval_raw = load_dataset("json", data_files="data/eval.jsonl", split="train")
    eval_ds  = eval_raw.map(fmt, remove_columns=[c for c in eval_raw.column_names if c != "text"])
except Exception as e:
    print(f"[warn] no eval set: {e}")

if eval_ds is not None:
    assert all(len(t) > 0 for t in eval_ds["text"]), "Found empty text rows in eval set"
'''