'''

python scripts/data_prep/merge_and_split.py \
    data/train.clean.jsonl data/train.follow3m.clean.jsonl

python scripts/data_prep/merge_and_split.py \
    data/claude_logs/train.clean.jsonl \
    data/claude_logs/train.follow3m.clean.jsonl \
    data/actual_code/train.fim.clean.jsonl

python scripts/data_prep/merge_and_split.py \
    data/claude_logs/train.clean.jsonl \
    data/claude_logs/train.follow3m.clean.jsonl \
    data/actual_code/train.fim.clean.jsonl \
    --name full \
--eval_frac 0.15

python scripts/data_prep/merge_and_split.py \
    data/train.full.jsonl \
    data/git_history/git_sft.clean.jsonl \
    data/actual_code/train.fim.clean.jsonl \
    --name full_v2 \
    --eval_frac 0.15
# Result: data/train.homematch_v2.jsonl (and no eval)
cp data/eval.full.jsonl data/eval.full_v2.jsonl

'''
# scripts/data_prep/merge_and_split.py
import argparse, json, random, hashlib
from pathlib import Path
from datetime import datetime

def sig_of(msgs):
    s = "\n".join(f"{(m.get('role') or '').lower()}:{(m.get('content') or '').strip()}" for m in msgs)
    return hashlib.md5(s.encode("utf-8","ignore")).hexdigest()

def read_jsonl(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                msgs = obj.get("messages")
                if msgs: yield msgs
            except Exception:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="clean JSONL files to merge")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--name", default=None, help="suffix for output files, e.g. 'full' -> train.full.jsonl")
    ap.add_argument("--eval_frac", type=float, default=None, help="fraction for eval split, e.g. 0.15")
    ap.add_argument("--eval_size", type=int, default=None, help="absolute eval size (overrides eval_frac)")
    ap.add_argument("--min_eval", type=int, default=100, help="minimum eval size if possible")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite outputs if they exist")
    args = ap.parse_args()

    # Collect + dedupe
    seen, all_msgs = set(), []
    per_file_counts, dup_count = {}, 0
    for p in args.inputs:
        c = 0
        for msgs in read_jsonl(p):
            c += 1
            h = sig_of(msgs)
            if h in seen:
                dup_count += 1
                continue
            seen.add(h)
            all_msgs.append(msgs)
        per_file_counts[p] = c

    total = len(all_msgs)
    random.seed(args.seed)
    random.shuffle(all_msgs)

    # Decide eval size
    if args.eval_size is not None:
        eval_sz = min(args.eval_size, max(0, total-1))
    else:
        frac = args.eval_frac if args.eval_frac is not None else 0.15
        eval_sz = int(round(total * frac))
        eval_sz = max(min(eval_sz, total//2), 0)  # cap at half, never negative
        eval_sz = max(eval_sz, min(args.min_eval, max(0, total-1))) if total >= args.min_eval else max(1, eval_sz)

    eval_sz = min(eval_sz, total-1) if total > 1 else 0  # keep at least 1 for train if possible
    eval_set = all_msgs[:eval_sz]
    train_set = all_msgs[eval_sz:]

    # Output names
    suffix = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"train.{suffix}.jsonl"
    eval_path  = out_dir / f"eval.{suffix}.jsonl"

    if not args.force:
        for p in (train_path, eval_path):
            if p.exists():
                raise SystemExit(f"Refusing to overwrite {p} (use --force).")

    # Write
    def write(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for msgs in items:
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    write(train_path, train_set)
    write(eval_path,  eval_set)

    # Report
    print("=== Merge & Split Report ===")
    for k,v in per_file_counts.items():
        print(f"  {k}: {v} lines")
    print(f"Unique after dedupe: {total}  (duplicates skipped: {dup_count})")
    print(f"Train: {len(train_set)}  Eval: {len(eval_set)}  (suffix='{suffix}')")
    print(f"Wrote → {train_path} and {eval_path}")

if __name__ == "__main__":
    main()
