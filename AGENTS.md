# Repository Guidelines

This guide helps contributors work productively in this repo. Keep changes focused, reproducible, and aligned with the structure and tooling below.

## Project Structure & Module Organization
- `scripts/`: pipelines and utilities
  - `data_prep/`: logs → SFT, repos → FIM, merge/split tools.
  - `training/`: QLoRA trainers, LoRA merge.
  - `testing/`: baseline checks, comparisons, perf/model tests.
  - `serving/` (optional): basic web UI.
- `data/`: datasets (gitignored). Use `data/staging/` for small samples; final splits in `data/train.jsonl`, `data/eval.jsonl`.
- `outputs/`, `runs/`, `logs/`: checkpoints, TensorBoard, run logs (gitignored).
- `pyproject.toml`: Python 3.12 deps; CUDA 11.8 wheels via `[tool.uv]`.

## Build, Test, and Development Commands
- Env setup: `uv sync && source .venv/bin/activate`.
- Data prep:
  - `python scripts/data_prep/claude_logs/claude2sft.py --src data/claude_logs --out data/train.clean.jsonl`
  - `python scripts/data_prep/actual_code/repo2fim.py --repo /path/to/repo --out data/train.fim.jsonl`
  - `python scripts/data_prep/merge_and_split.py data/*.jsonl --name full --eval_frac 0.15`
- Train: `./scripts.sh train` (or `./scripts.sh tmux-train`).
- Monitor: `./scripts.sh tensorboard` (6006), `./scripts.sh gpu`, `./scripts.sh logs`.
- Model test: `./scripts.sh model-test --help` (choose 7b/14b/32b, 4bit/8bit, offload budgets).

## Coding Style & Naming Conventions
- Python PEP 8, 4-space indents. Add type hints for public functions and concise docstrings.
- Naming: files/modules `snake_case.py`; functions/vars `lower_snake`; constants `UPPER_SNAKE`.
- Formatting: keep diffs minimal; if used locally, prefer `black` + `isort` + `ruff` (no reformat-only commits).

## Testing Guidelines
- Tests are script-based under `scripts/testing/` (no pytest).
- Quick checks: `baseline.py`, `compare_baseline_vs_lora.py`, `check_lengths.py`.
- Performance: `model_testing.py` (tokens/sec, GPU/CPU memory). Example: `./scripts.sh model-test --models 7b 14b --quant auto`.
- Keep JSONL fixtures tiny in `data/staging/`; never commit large datasets.

## Commit & Pull Request Guidelines
- Commits: Conventional (`feat:`, `fix:`, `docs:`), imperative mood, ≤72-char subject, focused diffs.
- PRs: clear description, repro commands, linked issues, relevant logs/TensorBoard screenshots; update docs when behavior changes.
- Hygiene: do not commit artifacts, checkpoints, or secrets; respect `.gitignore`.

## Security & Configuration Tips
- Secrets: use local `.env`; never commit tokens.
- CUDA: target 11.8; verify drivers; set `CUDA_VISIBLE_DEVICES` as needed.
