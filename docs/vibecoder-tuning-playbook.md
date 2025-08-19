
# Vibecoder Tuning Playbook
_Last updated: 2025-08-17 16:24 (local)_

This playbook gathers **practical, GPU-friendly** methods to turn a base coding model (e.g., `Qwen/Qwen2.5-Coder-3B-Instruct`) into **your vibecoding companion** on a single 2080 Ti (11 GB). It includes **dataset recipes, training configs, preference tuning, outcome-based RL**, serving via **Ollama/MCP**, and **maintenance**.

---

## 0) Quick reference

**Recommended sequence**
1. **SFT** on your curated data (Claude logs + repo FIM).
2. **Preference tuning** (KTO/ORPO, then DPO) on your 👍/👎 logs.
3. **Rejection-Sampling SFT** (RFT): sample N, filter by tests/lint → SFT on winners.
4. Optional: **Tiny PPO** on a dockerized micro-suite (fast tests), 4-bit LoRA.
5. Serve with **Ollama** and wire to **MCP/zen-mcp** tools.
6. **Continual learning loop**: nightly data export → short refresh LoRA.

**GPU-safe defaults (2080 Ti)**
- Quant: **8-bit** (or **4-bit QLoRA** if tight)
- LoRA: `r=16–32`, `alpha=32–64`, `dropout=0.05`
- Length: `max_length=1024–1536`
- Batches: `per_device_train_batch_size=1–2` with `gradient_accumulation_steps=8–24`
- LR: `5e-6 .. 1e-5` (SFT/Pref), cosine, warmup `10–20%`
- Eval cadence: every `50–100` steps; `metric_for_best_model="eval_loss"`

---

## 1) Build datasets from Git history (repo-aware)

### 1.1 Commit-Diff SFT (before → after)
Turn small, self-contained commits into instruction-following examples.

**Extraction idea**
- Filter to **small commits** (≤400 changed lines, ≤3 files).
- Exclude vendored/build/lock files.
- Keep commits with messages that read like intents (“fix null user_id”, “add shadcn Card”).
- For each changed file: capture **before snippet**, **after snippet**, and **commit message**.

**SFT example (chat-style)**
```json
{
  "messages": [
    {"role": "system", "content": "You are a terse, patch-first coding assistant."},
    {"role": "user", "content": "Commit intent: fix KeyError: 'user_id' in auth pipeline. File: src/auth/session.tsx (excerpt)\n<BEFORE>...\n"},
    {"role": "assistant", "content": "```diff\n--- a/src/auth/session.tsx\n+++ b/src/auth/session.tsx\n@@\n- const id = session.user['user_id'];\n+ const id = session.user?.user_id ?? null;\n```\nReason: guard missing 'user_id'."}
  ]
}
```

### 1.2 FIM++ (Fill-in-the-Middle) from repo
- For each file, sample ~6 short spans: **prefix**, **middle (masked)**, **suffix**.
- Instruction: “Complete the missing region.”
- Great for familiarizing the model with **your** architecture/components/styles.

**FIM example**
```json
{
  "messages": [
    {"role":"system","content":"Fill the missing code exactly; minimal changes."},
    {"role":"user","content":"<PREFIX>... useClient() {\n  const [open, setOpen] = useState(false)\n  ...\n}\n<MASK/>\n<SUFFIX>return (<Dialog open={open} onOpenChange={setOpen}>...")},
    {"role":"assistant","content":"<MIDDLE>useEffect(() => { setOpen(props.defaultOpen ?? false) }, [props.defaultOpen])\n</MIDDLE>"}
  ]
}
```

### 1.3 Bug → Patch pairs (tests or trace)
Use failing test output + file excerpt → minimal diff.

**Spec**
- Input: traceback or `pytest` failure, and snippet of the suspected file.
- Output: smallest-applies diff + one-line rationale.

---

## 2) Preference datasets (for vibe/style)

### 2.1 Unary thumbs (KTO/ORPO-friendly)
Log `(messages, assistant, label)` where `label ∈ {0,1}` from your `/good` or `/bad` votes.

```json
{ "messages": [...], "assistant": "<final reply or patch>", "label": 1 }
```

### 2.2 Pairwise (DPO-friendly)
For each prompt signature, keep one **chosen** and one **rejected** reply.

```json
{ "messages": [...], "chosen": "<good>", "rejected": "<bad>" }
```

**Where they come from**
- IDE/CLI proxy that intercepts `/good` and `/bad` (SQLite backed).
- Rejection sampling: generate 3–5 candidates, pick winner via rules (tests/lint/shortest).

---

## 3) Training recipes (TRL 0.21, LoRA, 8/4-bit)

### 3.1 SFT (Supervised Fine-Tuning)
- **Goal:** Teach repo familiarity + baseline behaviors.
- **Config (safe on 11 GB):**
  - 8-bit or 4-bit base
  - `max_length=1024–1536`
  - `bs=1`, `grad_accum=16–24`
  - `lr=1e-5`, warmup `0.1`, cosine
  - `group_by_length=True`, `packing=False`
  - `metric_for_best_model="eval_loss"`, `greater_is_better=False`
  - `per_device_eval_batch_size=1`, `eval_accumulation_steps=8`

### 3.2 Preference tuning (KTO/ORPO/DPO)
- **KTO/ORPO (unary ok):**
  - start from your best SFT adapter; LoRA rank 16–32
  - `max_length=1024`, `bs=1–2`, `grad_accum=8–16`, `lr=5e-6..1e-5`, `1–2` epochs
- **DPO (pairwise):**
  - reference model optional (SimPO/ORPO variants skip it)
  - similar hyperparams as above; add `beta≈0.1`

**Mask to assistant tokens** with `DataCollatorForCompletionOnlyLM` to stabilize chat SFT.

### 3.3 Rejection-Sampling SFT (RFT)
- Generate N completions, **filter** with cheap heuristics:
  - tests pass, typecheck clean, eslint clean
  - patch size small, files touched ≤ 2
- SFT again on **winners only** for 1–2 epochs.

### 3.4 Tiny PPO (programmatic rewards) — optional
- **When:** after you have a small dockerized micro-suite (fast tests).  
- **Reward:** +1 if tests pass, +0.2 typecheck, +0.2 eslint; penalties for patch size, touched files, token count.  
- **Config:** 4-bit, `bs=1`, rollout steps 32–64, `max_new_tokens 128–256`, short prompts, KL coeff small.  
- **Scope:** a few hundred episodes; save adapters.

---

## 4) Git → Dataset scripts (sketches)

### 4.1 Extract commit diffs
Pseudocode outline (Python + gitpython):
```python
# pip install gitpython unidiff
from git import Repo
from unidiff import PatchSet
import json, os, re

repo = Repo('~/code/your_repo')
pairs = []
for c in repo.iter_commits('--no-merges', max_count=5000):
    stats = c.stats.total
    if stats['files'] > 3 or stats['lines'] > 400:
        continue
    msg = c.message.strip()
    if re.search(r'WIP|bump|chore|merge', msg, re.I): 
        continue

    parent = c.parents[0] if c.parents else None
    if not parent: continue
    diff = repo.git.diff(parent.hexsha, c.hexsha, unified=3, ignore_blank_lines=True, ignore_space_at_eol=True)
    patch = PatchSet(diff)
    for f in patch:
        if any(f.path.endswith(x) for x in ('.lock','.min.js','.map','.png','.jpg')): 
            continue
        before, after = [], []
        for h in f:
            for l in h.source: 
                if l.is_context or l.is_removed: before.append(l.value)
            for l in h.target: 
                if l.is_context or l.is_added: after.append(l.value)
        if before and after:
            pairs.append({
                "messages":[
                    {"role":"system","content":"You are a terse, patch-first coding assistant."},
                    {"role":"user","content":f"Commit: {msg}\nFile: {f.path}\n<BEFORE>\n{''.join(before[-200:])}\n"}
                ],
                "assistant":"```diff\n" + str(f) + "\n```"
            })
os.makedirs('data', exist_ok=True)
with open('data/git_sft.jsonl','w') as o:
    for p in pairs: o.write(json.dumps(p, ensure_ascii=False) + "\n")
```

### 4.2 Produce FIM spans
For each file, sample 6 spans: write `<PREFIX>…<MASK/>…<SUFFIX>` into the user message and the missing chunk as the assistant reply.

---

## 5) Serving locally (Ollama + MCP)

### 5.1 Merge & quantize for Ollama
- After SFT/preference, merge LoRA:
  ```python
  merged = model.merge_and_unload()
  merged.save_pretrained("outputs/vibecoder_merged_fp16")
  ```
- Quantize to GGUF (Q4_K_M / Q5_K_M) via llama.cpp tools.
- **Modelfile**
  ```
  FROM ./vibecoder.Q5_K_M.gguf
  PARAM stop "<|im_end|>"
  SYSTEM You are a terse, code-first pair programmer...
  TEMPLATE ... (match Qwen chat template you trained on) ...
  ```
- `ollama create vibecoder -f Modelfile` → `ollama run vibecoder`

### 5.2 MCP/zen-mcp
- Wrap tools you trained on: **fs read/write (patch), ripgrep, bash, http, tests**.
- Guardrails: file whitelist, max patch size, confirm multi-file changes.

---

## 6) Continual learning loop (nightly)

1. Capture logs: prompts, tool calls, patches, outcomes, 👍/👎.  
2. Export:
   - SFT delta (new high-quality traces)
   - Preference pairs (KTO/ORPO/DPO-ready)  
3. Train short LoRA refresh (≤1 epoch).  
4. Canary eval (20 prompts): must pass before promoting.  
5. Version adapters, push to Ollama.

---

## 7) Eval harness (keep yourself honest)

- **Static set** (20–50 prompts): FIM, stacktrace triage, shadcn component, Next.js route, TS type fix, jest fail → patch.
- **Metrics**: eval loss, pass@1 on tiny tests, length of diff, files touched.
- **Report** at each save step and at end.

---

## 8) Troubleshooting & performance

- **OOM at eval** → `per_device_eval_batch_size=1`, `eval_accumulation_steps=8`, `max_length=1536/1408`.
- **OOM at train** → `bs=1`, lower `max_length`, LoRA `r=16`, switch to **4-bit**.
- **Slow steps** → shorter `max_length` (1024), reduce eval cadence to 100, maybe `bs=2` with higher accum if VRAM allows.
- **Noisy early loss** → normal with warmup + length bucketing; judge by `eval_loss`.
- **Best model not loading** → `metric_for_best_model="eval_loss"` and make `save_steps` multiple of `eval_steps`.

---

## 9) Minimal configs

### 9.1 SFT (safe on 2080 Ti)
```python
SFTConfig(
  per_device_train_batch_size=1,
  gradient_accumulation_steps=24,
  max_length=1408,
  learning_rate=1e-5,
  warmup_ratio=0.1,
  lr_scheduler_type="cosine",
  weight_decay=0.05,
  fp16=True,
  optim="adamw_bnb_8bit",
  eval_strategy="steps", eval_steps=50,
  save_strategy="steps", save_steps=50,
  per_device_eval_batch_size=1, eval_accumulation_steps=8,
  group_by_length=True, packing=False,
  load_best_model_at_end=True, metric_for_best_model="eval_loss",
  greater_is_better=False,
)
```

### 9.2 Preference (ORPO/DPO/KTO)
```python
# similar but shorter length and 1 epoch
max_length=1024, num_train_epochs=1, lr=5e-6..1e-5,
per_device_train_batch_size=1–2, grad_accum=8–16
```

---

## 10) tmux + mobile monitoring

- Windows: `run`, `gpu`, `tb` windows
- Mobile: Tailscale/SSH or direct TB via `--bind_all :6006`
- Handy: `tmux list-windows`, `rename-window`, `kill-window`, `kill-session`

---

### Appendix: thumbs-up/downs → datasets

- **Unary (KTO/ORPO)**: `{ "messages": [...], "assistant": "...", "label": 1 }`
- **Pairwise (DPO)**: `{ "messages": [...], "chosen": "...", "rejected": "..." }`
- Use a tiny FastAPI proxy to intercept `/good` & `/bad` and log to SQLite; nightly script exports `data/kto.jsonl` and `data/dpo.jsonl`.

---

**Rule of thumb:** SFT for **facts**, DPO/ORPO for **vibes**, RFT/PPO for **outcomes**. Keep runs small, evaluate often, and loop weekly with your newest traces.
