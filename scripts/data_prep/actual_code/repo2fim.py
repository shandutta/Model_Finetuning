'''
python scripts/data_prep/actual_code/repo2fim.py \
  --repo /mnt/c/Coding/HomeMatch_VSCode/homematch-v2/src \
  --out data/train.fim.jsonl \
  --per_file 6 \
  --max_len 1600

'''

import argparse, os, json, random, re, hashlib, pathlib

EXTS = {".ts",".tsx",".js",".jsx",".py",".sql",".json",".css",".scss",".md"}

def read_text(p, max_bytes=800_000):
    try:
        with open(p,"r",encoding="utf-8",errors="ignore") as f:
            return f.read(max_bytes)
    except Exception:
        return ""

def chunk_fim(code, max_len=1200):
    # strip huge files, normalize newlines
    code = code.replace("\r\n","\n").replace("\r","\n")
    if len(code) < 200: return []
    # choose a window up to max_len
    if len(code) > max_len:
        start = random.randint(0, max(0, len(code)-max_len))
        code = code[start:start+max_len]
    if "\n" not in code: return []
    # pick a middle span (20%–40% of window)
    L = len(code)
    mid_len = max(80, int(L * random.uniform(0.2, 0.4)))
    if mid_len >= L-40: return []
    mid_start = random.randint(20, L - mid_len - 20)
    prefix = code[:mid_start]
    middle = code[mid_start:mid_start+mid_len]
    suffix = code[mid_start+mid_len:]
    return [(prefix, middle, suffix)]

def make_example(path, pref, mid, suff):
    # Chat-style FIM instruction
    user = (
        f"[FILE] {path}\n"
        "Fill in the missing code between <FILL> and </FILL>. "
        "Preserve formatting and indentation. Respond with code only.\n\n"
        "PREFIX:\n"
        f"{pref}\n"
        "<FILL>\n"
        "...\n"
        "</FILL>\n"
        "SUFFIX:\n"
        f"{suff}"
    )
    return {
        "messages":[
            {"role":"system","content":"You are a precise coding assistant that completes code given a prefix and suffix."},
            {"role":"user","content":user},
            {"role":"assistant","content":mid}
        ]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="data/train.fim.jsonl")
    ap.add_argument("--per_file", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written, seen = 0, set()

    with open(args.out, "w", encoding="utf-8") as out:
        for root,_,files in os.walk(args.repo):
            for name in files:
                if pathlib.Path(name).suffix.lower() not in EXTS: continue
                path = os.path.join(root, name)
                txt = read_text(path)
                if not txt: continue
                # simple filtering of minified/huge one-liners
                if len(txt) > 20_000 and "\n" not in txt[:2000]: continue

                for _ in range(args.per_file):
                    samples = chunk_fim(txt, max_len=args.max_len)
                    for pref, mid, suff in samples:
                        h = hashlib.md5((pref+"§"+mid+"§"+suff).encode("utf-8","ignore")).hexdigest()
                        if h in seen: continue
                        seen.add(h)
                        ex = make_example(path, pref, mid, suff)
                        out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        written += 1

    print(f"Wrote {written} FIM examples to {args.out}")

if __name__ == "__main__":
    main()
