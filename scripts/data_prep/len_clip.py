import sys, json
from transformers import AutoTokenizer

MID = "Qwen/Qwen2.5-Coder-3B-Instruct"

def build_text(tok, ex):
    # Prefer precomputed "text"; otherwise, build from "messages"
    if ex.get("text"):
        return ex["text"]
    msgs = ex.get("messages")
    if msgs:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    # Fallback: empty
    return ""

def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/data_prep/len_clip.py <in.jsonl> <out.jsonl> <cap_tokens> [truncate|drop]")
        sys.exit(1)

    inp, outp, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "truncate"  # or "drop"

    tok = AutoTokenizer.from_pretrained(MID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # not used, but safe default

    kept = clipped = dropped = 0

    with open(inp, "r", encoding="utf-8", errors="ignore") as f, \
         open(outp, "w", encoding="utf-8") as w:

        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)

            text = build_text(tok, ex)
            if text == "":
                # Nothing usable — skip
                dropped += 1
                continue

            ids = tok(text, add_special_tokens=False).input_ids

            if len(ids) <= cap:
                # Keep as-is, but ensure "text" exists to bypass re-templating later
                if "text" not in ex or ex["text"] != text:
                    ex["text"] = text
                w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1
            else:
                if mode == "drop":
                    dropped += 1
                    continue
                # Truncate and write a "text" field with the clipped string
                clipped_text = tok.decode(ids[:cap], skip_special_tokens=False)
                ex["text"] = clipped_text
                w.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1
                clipped += 1

    print(f"wrote={kept}  clipped={clipped}  dropped={dropped}  cap={cap}  mode={mode}")

if __name__ == "__main__":
    main()
