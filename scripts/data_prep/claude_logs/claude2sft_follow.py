'''This script processes Claude/Cline JSONL logs to create a dataset for fine-tuning language models.
Sample Tool Call:

python scripts/data_prep/claude2sft_follow.py \
  --src /mnt/c/Users/Shan/.claude/projects/C--Coding-HomeMatch-VSCode-homematch-v2 \
  --cwd_contains homematch-v2 \
  --out data/train.follow3m.jsonl \
  --follow_minutes 3 \
  --limit 50000 \
  --max_tool_chars 1500 \
  --min_user_chars 60
'''

# claude2sft_follow.py
import os, json, argparse
from tqdm import tqdm
from datetime import datetime, timedelta

KEEP_USER_SEG_TYPES = {"text", "input_text", "message", "input", "tool_result"}
ASSISTANT_TEXT_TYPES = {"text"}

SYS_PROMPT = (
  "You are a precise coding assistant that diagnoses errors step-by-step, "
  "references file paths/lines when useful, and proposes concrete fixes."
)

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try: yield json.loads(line)
                except Exception: continue

def _seg_text(seg):
    if isinstance(seg, str): return seg
    return seg.get("text") or seg.get("content")

def has_human_text(segments):
    if isinstance(segments, str): return True
    for seg in segments or []:
        if isinstance(seg, str): return True
        t = seg.get("type")
        if t in KEEP_USER_SEG_TYPES - {"tool_result"}:
            val = _seg_text(seg)
            if isinstance(val, str) and val.strip():
                return True
    return False

def human_only_text(segments):
    parts = []
    if isinstance(segments, str): return segments.strip()
    for seg in segments or []:
        if isinstance(seg, str):
            parts.append(seg); continue
        t = seg.get("type")
        if t in KEEP_USER_SEG_TYPES - {"tool_result"}:
            val = _seg_text(seg)
            if isinstance(val, str) and val.strip():
                parts.append(val)
    return "\n".join(parts).strip()

def extract_user_text(segments, max_tool_chars=4000):
    parts = []
    if isinstance(segments, str): return segments.strip()
    for seg in segments or []:
        if isinstance(seg, str):
            parts.append(seg); continue
        t = seg.get("type")
        if t not in KEEP_USER_SEG_TYPES: continue
        if t == "tool_result":
            val = seg.get("content")
            if isinstance(val, str) and val.strip():
                txt = "\n[TOOL RESULT]\n" + val
                if len(txt) > max_tool_chars:
                    txt = txt[:max_tool_chars] + "\n[...truncated tool output...]"
                parts.append(txt)
        else:
            val = _seg_text(seg)
            if isinstance(val, str) and val.strip():
                parts.append(val)
    return "\n".join(parts).strip()

def extract_assistant_text(segments):
    parts = []
    if isinstance(segments, str): return segments.strip()
    for seg in segments or []:
        if isinstance(seg, str):
            parts.append(seg); continue
        if seg.get("type") in ASSISTANT_TEXT_TYPES:
            val = _seg_text(seg)
            if isinstance(val, str) and val.strip():
                parts.append(val)
    return "\n".join(parts).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with Claude/Cline .jsonl logs (recurses)")
    ap.add_argument("--out", default="data/train.follow.jsonl", help="Output JSONL")
    ap.add_argument("--follow_k", type=int, default=50, help="Max subsequent events to include after a human turn")
    ap.add_argument("--follow_minutes", type=int, default=0, help="If >0, include events until anchor_time + N minutes (takes precedence over --follow_k)")
    ap.add_argument("--limit", type=int, default=4000, help="Max examples to write")
    ap.add_argument("--max_tool_chars", type=int, default=4000)
    ap.add_argument("--min_user_chars", type=int, default=80)
    ap.add_argument("--cwd_contains", default=None, help="Only keep records whose `cwd` contains this substring")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 1) collect per session
    sessions = {}
    files = []
    for root, _, fs in os.walk(args.src):
        for name in fs:
            if name.lower().endswith(".jsonl"):
                files.append(os.path.join(root, name))

    for p in tqdm(files, desc="Scanning logs"):
        for obj in iter_jsonl(p):
            if args.cwd_contains:
                cwd = obj.get("cwd") or ""
                if args.cwd_contains not in cwd:
                    continue
            sid = obj.get("sessionId") or obj.get("session_id") or "default"
            sessions.setdefault(sid, []).append(obj)

    for sid in list(sessions.keys()):
        sessions[sid].sort(key=lambda o: o.get("timestamp", ""))\
        
    def _parse_ts(s):
        if not s: return None
        try:
            # handle ...Z
            if s.endswith("Z"): s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        except Exception:
            return None

    # 2) build follow windows
    written = 0
    total_windows = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for sid, recs in tqdm(list(sessions.items()), desc="Building windows"):
            i = 0
            L = len(recs)
            while i < L and written < args.limit:
                obj = recs[i]
                msg = obj.get("message") or {}
                role = msg.get("role") or obj.get("type")
                segs = msg.get("content") or []

                # Anchor only on human text turns
                if role == "user" and has_human_text(segs):
                    anchor = human_only_text(segs)
                    if len(anchor) >= args.min_user_chars:
                        anchor_ts = _parse_ts(obj.get("timestamp"))
                        window_end = None
                        if args.follow_minutes and anchor_ts:
                            window_end = anchor_ts + timedelta(minutes=args.follow_minutes)
                        messages = [
                            {"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": anchor}
                        ]
                        follow = 0
                        j = i + 1
                        saw_assistant = False
                        while j < L and follow < args.follow_k:
                            nxt = recs[j]
                            n_ts = _parse_ts(nxt.get("timestamp"))
                            if window_end and (not n_ts or n_ts > window_end):
                                break 
                            nmsg = nxt.get("message") or {}
                            nrole = nmsg.get("role") or nxt.get("type")
                            nsegs = nmsg.get("content") or []

                            if nrole == "assistant":
                                atext = extract_assistant_text(nsegs)
                                if atext:
                                    messages.append({"role": "assistant", "content": atext})
                                    saw_assistant = True
                                    follow += 1
                            elif nrole == "user":
                                # stop if next human text appears
                                if has_human_text(nsegs):
                                    break
                                # but include tool_result as context
                                utool = extract_user_text(nsegs, args.max_tool_chars)
                                if utool:
                                    messages.append({"role": "user", "content": utool})
                                    follow += 1
                            j += 1

                        if saw_assistant:
                            out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                            written += 1
                            total_windows += 1
                        i = j  # jump to end of window (reduces overlap)
                        continue
                i += 1

    print(f"Wrote {written} examples to {args.out} (windows built: {total_windows})")

if __name__ == "__main__":
    main()
