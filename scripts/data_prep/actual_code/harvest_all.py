#!/usr/bin/env python3
"""
Harvest commit→patch SFT examples + FIM samples from two repos (paths or Git URLs).

- Clones URLs to a temp dir (shallow + blobless), or uses local paths.
- Traverses all branches (optionally including remotes) for SFT.
- Generates FIM data with your preferred schema (chat-style, prefix/middle/suffix).
- Appends both repos' outputs into the same SFT and FIM files (dedup-aware).

Usage (example):
source .venv/bin/activate

python scripts/data_prep/actual_code/harvest_all.py \
--repo-a https://github.com/shandutta/homematch-v2.git \
--repo-b https://github.com/shandutta/homematch.git \
--out-sft data/git_history/git_sft.jsonl \
--out-fim data/actual_code/train.fim.jsonl \
--fim-per-file 6 --fim-max-len 1600 \
--all-branches --include-remotes \
--max-commits 50000 --max-files 50 --max-lines 1200 --before-budget 8000
"""

import argparse, hashlib, json, os, random, re, shutil, subprocess, tempfile
from pathlib import Path
from typing import Iterable, Tuple, Optional, Set

# --- Git + diff deps ---
# pip install gitpython unidiff
from git import Repo
from unidiff import PatchSet

# -----------------------
# Helpers: paths & cloning
# -----------------------

def is_git_url(s: str) -> bool:
    return s.startswith(("http://","https://","git@"))

def clone_to_tmp(url: str, tmp_root: Path, depth: int = 300) -> Path:
    dst = tmp_root / (url.split("/")[-1].replace(".git","") or "repo")
    # fast shallow, blobless
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--depth", str(depth), url, str(dst)],
        check=True
    )
    # make sure we see remotes if requested later
    subprocess.run(["git", "-C", str(dst), "fetch", "--all", "--prune"], check=True)
    return dst

# -----------------------
# SFT (commit → patch) generator
# -----------------------

DEFAULT_SYS_PROMPT = "You are a terse, patch-first coding assistant."

SKIP_EXT = {
    ".lock", ".min.js", ".map", ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".ico", ".svg", ".pdf", ".ttf", ".otf", ".woff", ".woff2", ".zip", ".gz",
    ".bz2", ".7z", ".dylib", ".so", ".dll", ".exe", ".bin",
}

SKIP_DIRS = {
    "node_modules", "dist", "build", ".next", ".turbo", ".git", "out", "coverage",
    ".venv", "venv", "__pycache__", ".cache",
}

VENDOR_FILE_HINTS = re.compile(
    r"(package-lock\.json|pnpm-lock\.yaml|yarn\.lock|Cargo\.lock|Podfile\.lock|go\.sum|poetry\.lock|Gemfile\.lock)$",
    re.I,
)

INTENTLIKE_MSG = re.compile(
    r"(fix|add|update|refactor|clean|handle|guard|support|implement|remove|rename|migrate|speed|perf|docs?)",
    re.I,
)
BORING_MSG = re.compile(r"\b(WIP|bump|chore|merge|format|fmt|prettier)\b", re.I)

def should_skip_path(path: str) -> bool:
    p = Path(path)
    if any(part in SKIP_DIRS for part in p.parts):
        return True
    if p.suffix.lower() in SKIP_EXT:
        return True
    if VENDOR_FILE_HINTS.search(str(p)):
        return True
    return False

def last_n_chars(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[-n:]

def hunk_text_before_after(hunk) -> Tuple[str, str]:
    before_lines, after_lines = [], []
    for line in hunk:
        if line.is_context or line.is_removed:
            before_lines.append(line.value)
        if line.is_context or line.is_added:
            after_lines.append(line.value)
    return "".join(before_lines), "".join(after_lines)

def file_diff_to_patch_block(file_patch) -> str:
    return str(file_patch).rstrip()

def commit_is_small(stats_files: int, stats_lines: int, max_files: int, max_lines: int) -> bool:
    return (stats_files <= max_files) and (stats_lines <= max_lines)

def message_is_intentful(msg: str) -> bool:
    msg = msg.strip()
    if not msg:
        return False
    if BORING_MSG.search(msg):
        return False
    return bool(INTENTLIKE_MSG.search(msg))

def build_sft_example(commit_msg: str, file_path: str, before: str, patch_block: str,
                      before_char_budget: int, reason_from_subject: bool) -> dict:
    subject = commit_msg.splitlines()[0].strip()
    reason = subject if reason_from_subject else ""
    user_content = (
        f"Commit intent: {subject}\n"
        f"File: {file_path}\n"
        f"<BEFORE>\n{last_n_chars(before, before_char_budget)}"
    )
    assistant = f"```diff\n{patch_block}\n```"
    if reason:
        assistant += f"\nReason: {reason}"
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYS_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "assistant": assistant
    }

def _rev_iterables(repo: Repo, branch: str, all_branches: bool, include_remotes: bool):
    if all_branches:
        for h in repo.branches:
            yield h.name
        if include_remotes:
            for r in repo.remotes:
                for ref in r.refs:
                    if ref.name.endswith("/HEAD"):
                        continue
                    yield ref.name
    else:
        yield branch

def iter_sft_examples(repo: Repo,
                      max_commits: int,
                      max_files: int,
                      max_lines: int,
                      before_char_budget: int,
                      reason_from_subject: bool,
                      branch: str,
                      all_branches: bool,
                      include_remotes: bool) -> Iterable[dict]:
    seen = set()
    for rev in _rev_iterables(repo, branch, all_branches, include_remotes):
        for c in repo.iter_commits(rev, max_count=max_commits, no_merges=True):
            if c.hexsha in seen:
                continue
            seen.add(c.hexsha)

            stats = c.stats.total
            if not commit_is_small(stats["files"], stats["lines"], max_files, max_lines):
                continue

            msg = c.message.strip()
            if not message_is_intentful(msg):
                continue

            parent = c.parents[0] if c.parents else None
            if not parent:
                continue

            diff_text = repo.git.diff(
                parent.hexsha, c.hexsha,
                unified=3,
                ignore_blank_lines=True,
                ignore_space_at_eol=True,
            )
            if not diff_text.strip():
                continue

            try:
                patch = PatchSet(diff_text)
            except Exception:
                continue

            for f in patch:
                if should_skip_path(f.path):
                    continue

                before_concat, after_concat = [], []
                for h in f:
                    b, a = hunk_text_before_after(h)
                    if b.strip() or a.strip():
                        before_concat.append(b)
                        after_concat.append(a)
                if not before_concat or not after_concat:
                    continue

                before_text = "".join(before_concat)
                patch_block = file_diff_to_patch_block(f)
                if not patch_block.strip():
                    continue

                yield build_sft_example(
                    commit_msg=msg,
                    file_path=f.path,
                    before=before_text,
                    patch_block=patch_block,
                    before_char_budget=before_char_budget,
                    reason_from_subject=reason_from_subject
                )

def write_sft_for_repo(repo_path: Path, out_path: Path, args) -> int:
    repo = Repo(str(repo_path))
    # keep remotes fresh if requested
    if args.include_remotes:
        try:
            subprocess.run(["git", "-C", str(repo_path), "fetch", "--all", "--prune"], check=True)
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("a", encoding="utf-8") as w:
        for ex in iter_sft_examples(
            repo=repo,
            max_commits=args.max_commits,
            max_files=args.max_files,
            max_lines=args.max_lines,
            before_char_budget=args.before_budget,
            reason_from_subject=args.reason_from_subject,
            branch=args.branch,
            all_branches=args.all_branches,
            include_remotes=args.include_remotes,
        ):
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
    return count

# -----------------------
# FIM generator (your schema)
# -----------------------

FIM_EXTS = {".ts",".tsx",".js",".jsx",".py",".sql",".json",".css",".scss",".md"}

def read_text(p: Path, max_bytes=800_000) -> str:
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes)
    except Exception:
        return ""

def chunk_fim(code: str, max_len=1200):
    code = code.replace("\r\n","\n").replace("\r","\n")
    if len(code) < 200: return []
    if len(code) > max_len:
        start = random.randint(0, max(0, len(code)-max_len))
        code = code[start:start+max_len]
    if "\n" not in code: return []
    L = len(code)
    mid_len = max(80, int(L * random.uniform(0.2, 0.4)))
    if mid_len >= L-40: return []
    mid_start = random.randint(20, L - mid_len - 20)
    prefix = code[:mid_start]
    middle = code[mid_start:mid_start+mid_len]
    suffix = code[mid_start+mid_len:]
    return [(prefix, middle, suffix)]

def make_fim_example(path: str, pref: str, mid: str, suff: str) -> dict:
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

def write_fim_for_repo(repo_path: Path, out_path: Path, per_file: int, max_len: int,
                       rng_seed: int, dedupe_set: Set[str]) -> int:
    random.seed(rng_seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("a", encoding="utf-8") as out:
        for p in repo_path.rglob("*"):
            if not p.is_file(): continue
            if p.suffix.lower() not in FIM_EXTS: continue
            txt = read_text(p)
            if not txt: continue
            # filter: huge minified one-liners
            if len(txt) > 20_000 and "\n" not in txt[:2000]: continue

            for _ in range(per_file):
                samples = chunk_fim(txt, max_len=max_len)
                for pref, mid, suff in samples:
                    h = hashlib.md5((pref+"§"+mid+"§"+suff).encode("utf-8","ignore")).hexdigest()
                    if h in dedupe_set: continue
                    dedupe_set.add(h)
                    ex = make_fim_example(str(p), pref, mid, suff)
                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    written += 1
    return written

# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Harvest SFT (git) + FIM from two repos (paths or Git URLs).")
    # Repos
    ap.add_argument("--repo-a", required=True, help="Path or Git URL (e.g., homematch-v2)")
    ap.add_argument("--repo-b", required=True, help="Path or Git URL (e.g., legacy homematch)")
    # SFT knobs
    ap.add_argument("--out-sft", default="data/all/git_sft.jsonl")
    ap.add_argument("--branch", type=str, default="HEAD", help="Default branch/rev if not walking all branches")
    ap.add_argument("--max-commits", type=int, default=50000)
    ap.add_argument("--max-files", type=int, default=8)
    ap.add_argument("--max-lines", type=int, default=600)
    ap.add_argument("--before-budget", type=int, default=6000)
    ap.add_argument("--reason-from-subject", action="store_true")
    ap.add_argument("--all-branches", action="store_true")
    ap.add_argument("--include-remotes", action="store_true")
    # FIM knobs
    ap.add_argument("--out-fim", default="data/all/train.fim.jsonl")
    ap.add_argument("--fim-per-file", type=int, default=6)
    ap.add_argument("--fim-max-len", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=42)
    # clone depth
    ap.add_argument("--clone-depth", type=int, default=300)
    args = ap.parse_args()

    tmp_root = Path(tempfile.mkdtemp(prefix="harvest_all_"))
    clones = []
    try:
        # Resolve/clone repo A
        repo_a_path = None
        if is_git_url(args.repo_a):
            repo_a_path = clone_to_tmp(args.repo_a, tmp_root, depth=args.clone_depth)
            clones.append(repo_a_path)
        else:
            repo_a_path = Path(args.repo_a).expanduser().resolve()

        # Resolve/clone repo B
        repo_b_path = None
        if is_git_url(args.repo_b):
            repo_b_path = clone_to_tmp(args.repo_b, tmp_root, depth=args.clone_depth)
            clones.append(repo_b_path)
        else:
            repo_b_path = Path(args.repo_b).expanduser().resolve()

        # --- SFT: repo A then B (append) ---
        sft_out = Path(args.out_sft)
        total_sft = 0
        total_sft += write_sft_for_repo(repo_a_path, sft_out, args)
        total_sft += write_sft_for_repo(repo_b_path, sft_out, args)

        # --- FIM: repo A then B (append, dedupe by content hash) ---
        fim_out = Path(args.out_fim)
        fim_seen: Set[str] = set()
        total_fim = 0
        total_fim += write_fim_for_repo(repo_a_path, fim_out, args.fim_per_file, args.fim_max_len, args.seed, fim_seen)
        total_fim += write_fim_for_repo(repo_b_path, fim_out, args.fim_per_file, args.fim_max_len, args.seed+1, fim_seen)

        print(f"Wrote {total_sft} SFT examples → {sft_out}")
        print(f"Wrote {total_fim} FIM examples → {fim_out}")

    finally:
        # clean up clones
        for c in clones:
            try:
                shutil.rmtree(c, ignore_errors=True)
            except Exception:
                pass
        # Remove tmp_root if empty
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
