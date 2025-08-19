#!/usr/bin/env python3
"""
Create chat-style SFT examples from Git history (commit-diff pairs).

Output: JSONL at data/git_history/git_sft.jsonl
Each line is:
{
  "messages": [
    {"role":"system","content":"You are a terse, patch-first coding assistant."},
    {"role":"user","content":"Commit intent: ...\nFile: ...\n<BEFORE>\n..."},
  ],
  "assistant": "```diff\n--- a/path\n+++ b/path\n@@ ...\n- old\n+ new\n```\nReason: <commit subject>"
}
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Tuple

# pip install gitpython unidiff
from git import Repo
from unidiff import PatchSet

DEFAULT_SYS_PROMPT = "You are a terse, patch-first coding assistant."

# Extensions and paths we generally want to skip
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
    """
    Build 'before' and 'after' excerpts from a unidiff hunk.
    Keep context+removed for before, context+added for after.
    """
    before_lines, after_lines = [], []
    for line in hunk:
        if line.is_context or line.is_removed:
            before_lines.append(line.value)
        if line.is_context or line.is_added:
            after_lines.append(line.value)
    return "".join(before_lines), "".join(after_lines)


def file_diff_to_patch_block(file_patch) -> str:
    # unidiff's str(file_patch) yields a unified diff for that file
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


def build_example(commit_msg: str, file_path: str, before: str, patch_block: str, before_char_budget: int, reason_from_subject: bool) -> dict:
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
            {"role": "user", "content": user_content}
        ],
        "assistant": assistant
    }

def iter_examples(repo: Repo,
                  max_commits: int,
                  max_files: int,
                  max_lines: int,
                  before_char_budget: int,
                  reason_from_subject: bool,
                  branch: str,
                  all_branches: bool = False,
                  include_remotes: bool = False) -> Iterable[dict]:

    seen_commits = set()
    for rev in _rev_iterables(repo, branch, all_branches, include_remotes):
        for c in repo.iter_commits(rev, max_count=max_commits, no_merges=True):
            if c.hexsha in seen_commits:
                continue
            seen_commits.add(c.hexsha)

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

                yield build_example(
                    commit_msg=msg,
                    file_path=f.path,
                    before=before_text,
                    patch_block=patch_block,
                    before_char_budget=before_char_budget,
                    reason_from_subject=reason_from_subject
                )

def _rev_iterables(repo: Repo, branch: str, all_branches: bool, include_remotes: bool):
    """
    Yield revspecs to feed into repo.iter_commits().
    - If all_branches: yield each local branch name.
    - If include_remotes: also yield each remote/<name> branch.
    - Else: yield the single 'branch' argument (default HEAD/main).
    """
    if all_branches:
        for h in repo.branches:               # local branches
            yield h.name
        if include_remotes:
            for r in repo.remotes:
                for ref in r.refs:            # e.g. origin/main, origin/feature/x
                    if ref.name.endswith("/HEAD"):  # skip symbolic HEAD
                        continue
                    yield ref.name
    else:
        yield branch

def main():
    ap = argparse.ArgumentParser(description="Create SFT dataset from Git commit diffs.")
    ap.add_argument("--repo", type=str, default=".", help="Path to the Git repo")
    ap.add_argument("--branch", type=str, default="HEAD", help="Branch or rev range (e.g., main)")
    ap.add_argument("--max-commits", type=int, default=100000)
    ap.add_argument("--max-files", type=int, default=20, help="Skip commits touching > this many files")
    ap.add_argument("--max-lines", type=int, default=1200, help="Skip commits with > this many changed lines")
    ap.add_argument("--before-budget", type=int, default=8000, help="How many chars of <BEFORE> to include")
    ap.add_argument("--out", type=str, default="data/git_history/git_sft.jsonl")
    ap.add_argument("--reason-from-subject", action="store_true", help="Append 'Reason: <subject>' after diff")
    ap.add_argument("--all-branches", action="store_true", help="Traverse all local branches (de-dup commits).")
    ap.add_argument("--include-remotes", action="store_true", help="Include remote branches too (fetch first for freshness).")
    args = ap.parse_args()

    repo = Repo(args.repo)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as w:
        for ex in iter_examples(
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

    print(f"Wrote {count} examples → {out_path}")


if __name__ == "__main__":
    main()
