
import os, json, argparse
from tqdm import tqdm

KEEP_USER_SEG_TYPES = {"text", "input_text", "message", "input", "tool_result"}
ASSISTANT_TEXT_TYPES = {"text"}

def extract_user_text(segments, max_tool_chars=6000):
    parts = []
    if isinstance(segments, str):
        segments = [{"type":"text","text":segments}]
    for seg in segments or []:
        if isinstance(seg, str):
            parts.append(seg)
            continue
        t = seg.get("type")
        if t not in KEEP_USER_SEG_TYPES:
            continue
        if t == "tool_result":
            val = seg.get("content")
            if isinstance(val, str) and val.strip():
                txt = "\n[TOOL RESULT]\n" + val
                if len(txt) > max_tool_chars:
                    txt = txt[:max_tool_chars] + "\n[...truncated tool output...]"
                parts.append(txt)
        else:
            val = seg.get("text") or seg.get("content")
            if isinstance(val, str) and val.strip():
                parts.append(val)
    return "\n".join(parts).strip()

def extract_assistant_text(segments):
    parts = []
    if isinstance(segments, str):
        segments = [{"type":"text","text":segments}]
    for seg in segments or []:
        if isinstance(seg, str):
            parts.append(seg)
            continue
        if seg.get("type") in ASSISTANT_TEXT_TYPES:
            val = seg.get("text") or seg.get("content")
            if isinstance(val, str) and val.strip():
                parts.append(val)
    return "\n".join(parts).strip()

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing Claude/Cline .jsonl logs (recurses)")
    ap.add_argument("--out", default="data/train.jsonl", help="Output JSONL for SFT")
    ap.add_argument("--limit", type=int, default=100000, help="Max examples to write")
    ap.add_argument("--max_tool_chars", type=int, default=6000, help="Truncate tool_result context")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # collect all .jsonl paths first so tqdm can show progress
    all_files = []
    for root, _, files in os.walk(args.src):
        for name in files:
            if name.lower().endswith(".jsonl"):
                all_files.append(os.path.join(root, name))

    sessions = {}
    for p in tqdm(all_files, desc="Scanning logs", unit="file"):
        for obj in iter_jsonl(p):
                sid = obj.get("sessionId") or obj.get("session_id") or "default"
                sessions.setdefault(sid, []).append(obj)

    for sid in list(sessions.keys()):
        sessions[sid].sort(key=lambda o: o.get("timestamp", ""))

    written = 0
    from tqdm import tqdm as _tqdm
    with open(args.out, "w", encoding="utf-8") as out, _tqdm(total=args.limit, desc="Pairs", unit="pair") as pbar:
        for sid, recs in sessions.items():
            pending_user = None
            for obj in recs:
                typ = obj.get("type")
                msg = obj.get("message") or {}
                role = msg.get("role") or typ
                segments = msg.get("content") or []

                if role == "user":
                    text = extract_user_text(segments, args.max_tool_chars)
                    if not text:
                        tur = obj.get("toolUseResult")
                        if isinstance(tur, dict):
                            if "file" in tur and isinstance(tur["file"], dict):
                                c = tur["file"].get("content")
                                if isinstance(c, str) and c.strip():
                                    text = "[TOOL RESULT]\n" + c
                            elif isinstance(tur.get("text"), str):
                                text = "[TOOL RESULT]\n" + tur["text"]
                    if text:
                        pending_user = (pending_user + "\n\n" + text) if pending_user else text

                elif role == "assistant":
                    text = extract_assistant_text(segments)
                    if text and pending_user:
                        example = {
                            "messages": [
                                {"role": "system", "content": "You are a precise coding assistant that diagnoses errors step-by-step, references file paths/lines when useful, and proposes concrete fixes."},
                                {"role": "user", "content": pending_user.strip()},
                                {"role": "assistant", "content": text}
                            ]
                        }
                        out.write(json.dumps(example, ensure_ascii=False) + "\n")
                        written += 1
                        pbar.update(1)
                        pending_user = None
                        if written >= args.limit:
                            break
            if written >= args.limit:
                break

    print(f"Wrote {written} examples to {args.out}")

if __name__ == "__main__":
    main()
