# CLAUDE.md

QLoRA fine-tuning pipeline for Qwen/Qwen2.5-Coder-3B-Instruct model.

## Project Overview
Fine-tune coding assistants using QLoRA on multiple data sources (Claude logs, git history, FIM examples).

## Key Scripts
- **Training**: `scripts/training/sft_qlora_8bit.py` (production), `sft_qlora_4bit.py` (dev)
- **Data prep**: `scripts/data_prep/claude_logs/claude2sft.py`, `actual_code/repo2fim.py`, `merge_and_split.py`
- **Testing**: `scripts/testing/baseline.py`, `compare_baseline_vs_lora.py`

## Quick Commands
```bash
# Train (8-bit production)
python scripts/training/sft_qlora_8bit.py

# Monitor
tensorboard --logdir runs/qwen3b_8bit --port 6006
watch -n 1 nvidia-smi

# Prepare data
python scripts/data_prep/merge_and_split.py data/*.jsonl --name full --eval_frac 0.15
```

## Current Config (Polish Phase)
- **Learning rate**: 1.5e-5 (cosine, warmup=0.005)
- **LoRA**: r=32, α=64, dropout=0.05
- **Batch**: effective=24 (gradient accumulation)
- **Checkpoints**: Resume from `outputs/qwen3b_lora_8bit/checkpoint-1450`
- **Output**: `outputs/qwen3b_lora_8bit_polish/`

## Data Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a precise coding assistant..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Important Paths
- **Data**: `data/train.full.jsonl`, `data/eval.full.jsonl`
- **Outputs**: `outputs/qwen3b_lora_8bit_polish/`
- **Logs**: `runs/qwen3b_8bit/`, `logs/`