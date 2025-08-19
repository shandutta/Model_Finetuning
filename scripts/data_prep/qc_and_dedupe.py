'''
Sample Tool Call
python scripts/data_prep/qc_and_dedupe.py \
  --in data/train.follow3m.jsonl \
  --out data/train.follow3m.clean.jsonl \
  --min_user_chars 80 \
  --max_tokens 4096

'''

# qc_and_dedupe.py
import argparse, json, hashlib, textwrap
from collections import Counter
from tqdm import tqdm

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()

def join_msgs(msgs):
    # normalize for dedupe: role:content lines
    parts = []
    for m in msgs:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        parts.append(f"{role}:{content}")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--min_user_chars", type=int, default=80)
    ap.add_argument("--max_tokens", type=int, default=4096)
    args = ap.parse_args()

    # Try tokenizer for real token counts; fallback to char-based estimate
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct", trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    except Exception:
        pass

    total = 0
    kept = 0
    seen = set()
    drops = Counter()
    token_hist = []

    with open(args.inp, "r", encoding="utf-8", errors="ignore") as f, \
         open(args.out, "w", encoding="utf-8") as out:
        for line in tqdm(f, desc="QC"):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                drops["bad_json"] += 1
                continue

            msgs = obj.get("messages") or []
            if not msgs or not any(m.get("role")=="assistant" for m in msgs):
                drops["no_assistant"] += 1
                continue

            # anchor user (first user in window)
            user = next((m for m in msgs if m.get("role")=="user"), None)
            if not user or not (user.get("content") or "").strip():
                drops["no_user"] += 1
                continue
            if len((user.get("content") or "").strip()) < args.min_user_chars:
                drops["short_user"] += 1
                continue

            # dedupe
            sig = md5(join_msgs(msgs).lower())
            if sig in seen:
                drops["duplicate"] += 1
                continue
            seen.add(sig)

            # token length (approx if no tokenizer)
            if tok:
                try:
                    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    n_tok = len(tok(txt, return_tensors="pt").input_ids[0])
                except Exception:
                    txt = "\n".join([(m.get("content") or "") for m in msgs])
                    n_tok = int(len(txt) / 4)  # crude
            else:
                txt = "\n".join([(m.get("content") or "") for m in msgs])
                n_tok = int(len(txt) / 4)

            if n_tok > args.max_tokens:
                drops["too_long"] += 1
                continue

            token_hist.append(n_tok)
            out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            kept += 1

    def pct(n): 
        return f"{(100.0*n/total):.1f}%" if total else "0.0%"

    token_hist.sort()
    def q(p):
        if not token_hist: return 0
        i = int(p*(len(token_hist)-1))
        return token_hist[i]

    print("\n=== QC Summary ===")
    print(f"Total read:   {total}")
    print(f"Kept:         {kept}")
    print("Dropped:")
    for k,v in drops.most_common():
        print(f"  {k:>12}: {v:4d}  ({pct(v)})")
    if token_hist:
        print("\nToken stats (kept):")
        print(f"  p50: {q(0.50)}  p75: {q(0.75)}  p90: {q(0.90)}  p95: {q(0.95)}  max: {max(token_hist)}")
        print(f"  mean≈ {int(sum(token_hist)/len(token_hist))}")
        print(f"\nWrote cleaned file → {args.out}")

if __name__ == "__main__":
    main()
