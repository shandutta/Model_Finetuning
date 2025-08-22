# Repository Guidelines

## Project Structure & Module Organization
- `scripts/`: pipeline code
  - `data_prep/`: logs → SFT, repos → FIM, merge/split utilities.
  - `training/`: QLoRA trainers and LoRA merge tools.
  - `testing/`: baseline checks, comparisons, perf/model tests.
  - `serving/` (optional): simple web UI setup.
- `data/`: datasets (gitignored). Use `data/staging/` for small samples; final splits live in `data/train.jsonl` and `data/eval.jsonl`.
- `outputs/`, `runs/`, `logs/`: checkpoints, TensorBoard, and run logs (all gitignored).
- `pyproject.toml`: Python 3.12 deps; CUDA 11.8 wheels via `[tool.uv]`.

## Build, Test, and Development Commands
- Environment: `uv sync && source .venv/bin/activate`.
- Data prep examples:
  - `python scripts/data_prep/claude_logs/claude2sft.py --src data/claude_logs --out data/train.clean.jsonl`
  - `python scripts/data_prep/actual_code/repo2fim.py --repo /path/to/repo --out data/train.fim.jsonl`
  - `python scripts/data_prep/merge_and_split.py data/*.jsonl --name full --eval_frac 0.15`
- Train: `./scripts.sh train` (or `./scripts.sh tmux-train`).
- Monitor: `./scripts.sh tensorboard` (port 6006), `./scripts.sh gpu`, `./scripts.sh logs`.
- Model test: `./scripts.sh model-test --help` (choose 7b/14b/32b, 4bit/8bit, offload budgets).

## Coding Style & Naming Conventions
- Python PEP 8; 4-space indents. Use type hints for public functions and concise docstrings.
- Naming: files/modules `snake_case.py`; functions/vars `lower_snake`; constants `UPPER_SNAKE`.
- Formatting: no enforced tool; if used locally prefer `black` + `isort` + `ruff`. Avoid reformat-only changes.

## Testing Guidelines
- Tests are script-based under `scripts/testing/` (no pytest). Quick checks: `baseline.py`, `compare_baseline_vs_lora.py`, `check_lengths.py`.
- Performance: `model_testing.py` (tokens/sec, GPU/CPU memory). Example: `./scripts.sh model-test --models 7b 14b --quant auto`.
- Keep JSONL fixtures small in `data/staging/`; do not commit large datasets.

## Commit & Pull Request Guidelines
- Commits: Conventional style (`feat:`, `fix:`, `docs:`). Imperative mood, ≤72-char subject, focused diffs.
- PRs: clear description, reproduction commands, linked issues, relevant logs/TensorBoard screenshots, and updated docs when behavior changes.
- Hygiene: never commit artifacts or secrets; respect `.gitignore` for `data/`, `outputs/`, `runs/`, `logs/`.

## Security & Configuration Tips
- Secrets: use local `.env`; never commit tokens.
- CUDA: target CUDA 11.8 (e.g., RTX 2080 Ti). Ensure compatible drivers; set `CUDA_VISIBLE_DEVICES` when needed.

