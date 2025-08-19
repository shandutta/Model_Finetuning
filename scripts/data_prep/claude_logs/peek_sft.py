'''

Sample Tool Call
python peek_sft.py --path data/train.follow5m.jsonl --n 3 --max_chars 600
(optional) filter to examples where your anchor mentions "PropertySwiper":
python peek_sft.py --path data/train.follow5m.jsonl --contains PropertySwiper

'''

# peek_sft.py
import argparse, json, random, textwrap

def shorten(s, n):
    s = (s or "").strip()
    return textwrap.shorten(s, width=n, placeholder=" …")

def load_examples(path, contains=None):
    ex = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msgs = obj.get("messages") or []
            if contains:
                # only keep if the anchor user message contains the substring
                u = next((m for m in msgs if m.get("role")=="user"), None)
                if not u or contains.lower() not in (u.get("content") or "").lower():
                    continue
            ex.append(msgs)
    return ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="JSONL file (e.g., data/train.follow5m.jsonl)")
    ap.add_argument("--n", type=int, default=3, help="How many examples to show")
    ap.add_argument("--max_chars", type=int, default=600, help="Max chars to show per field")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contains", default=None, help="Filter for user text containing this substring")
    args = ap.parse_args()

    msgs_list = load_examples(args.path, args.contains)
    if not msgs_list:
        print("No examples found.")
        return

    random.seed(args.seed)
    picks = random.sample(msgs_list, k=min(args.n, len(msgs_list)))

    for i, msgs in enumerate(picks, 1):
        user = next((m for m in msgs if m.get("role")=="user"), None)
        asst = next((m for m in msgs if m.get("role")=="assistant"), None)

        print("="*88)
        print(f"Example {i}  |  turns={len(msgs)}  |  extra_turns={max(0, len(msgs)-3)}")
        print("-"*88)
        if user:
            print("USER ▶")
            print(shorten(user.get("content",""), args.max_chars))
            print()
        if asst:
            print("ASSISTANT ▶")
            print(shorten(asst.get("content",""), args.max_chars))
            print()
        # If you want to peek at follow-up turns, uncomment below:
        # for t, m in enumerate(msgs[ msgs.index(asst)+1 : msgs.index(asst)+3 ], 1):
        #     print(f"[FOLLOW {t}] {m.get('role','?').upper()} ▶")
        #     print(shorten(m.get('content',''), args.max_chars)); print()

if __name__ == "__main__":
    main()
