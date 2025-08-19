# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a model fine-tuning repository focused on training coding assistants using QLoRA (Quantized Low-Rank Adaptation) on the Qwen/Qwen2.5-Coder-3B-Instruct model. The project processes data from multiple sources (Claude logs, actual code) and trains models to improve code completion and debugging capabilities.

## Architecture

### Data Pipeline
- **Claude logs processing**: `scripts/data_prep/claude_logs/claude2sft.py` converts Claude conversation logs into SFT format
- **Code FIM generation**: `scripts/data_prep/actual_code/repo2fim.py` creates Fill-in-the-Middle examples from codebases
- **Data merging**: `scripts/data_prep/merge_and_split.py` combines multiple JSONL files, deduplicates, and creates train/eval splits
- **Quality control**: `scripts/data_prep/qc_and_dedupe.py` for data cleaning

### Training Infrastructure
- **QLoRA training**: Uses 4-bit and 8-bit quantization with LoRA adapters for memory efficiency
- **Model**: Qwen/Qwen2.5-Coder-3B-Instruct base model
- **Training scripts**: `scripts/training/sft_qlora_8bit.py` (production) and `scripts/training/sft_qlora_4bit.py` (development)

### Data Sources
- `data/claude_logs/`: Processed Claude conversation logs
- `data/actual_code/`: FIM examples from real codebases
- Final datasets: `data/train.full.jsonl` and `data/eval.full.jsonl`

## Common Commands

### Data Preparation
```bash
# Convert Claude logs to SFT format
python scripts/data_prep/claude_logs/claude2sft.py --src data/claude_logs/ --out data/train.clean.jsonl

# Generate FIM examples from codebase
python scripts/data_prep/actual_code/repo2fim.py --repo /path/to/repo --out data/train.fim.jsonl --per_file 6 --max_len 1600

# Merge datasets and create train/eval split
python scripts/data_prep/merge_and_split.py data/claude_logs/train.clean.jsonl data/claude_logs/train.follow3m.clean.jsonl data/actual_code/train.fim.clean.jsonl --name full --eval_frac 0.15
```

### Training
```bash
# 8-bit training (production)
python scripts/training/sft_qlora_8bit.py

# 4-bit training (development/testing)
python scripts/training/sft_qlora_4bit.py
```

### Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Start TensorBoard
tensorboard --logdir runs/qwen3b_8bit --port 6006

# Check dataset token length distribution
python check_lengths.py
```

### Testing Baseline Model
```bash
python baseline.py
```

## Key Configuration Details

### Training Parameters (8-bit)
- **Model**: Qwen/Qwen2.5-Coder-3B-Instruct with 8-bit quantization
- **LoRA**: r=32, alpha=64, dropout=0.05, targets=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
- **Batch size**: 1 per device, gradient accumulation=24 (effective batch=24)
- **Learning rate**: 2e-5 with cosine scheduler, warmup_ratio=0.10
- **Max length**: 3600 tokens (captures 95% of data)
- **Epochs**: 5 with early stopping (patience=5, threshold=0.002)

### Data Format
Training data uses chat format with system message:
```json
{
  "messages": [
    {"role": "system", "content": "You are a precise coding assistant that diagnoses errors step-by-step, references file paths/lines when useful, and proposes concrete fixes."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Important Paths
- **Outputs**: `outputs/qwen3b_lora_8bit/` (model checkpoints and adapters)
- **TensorBoard logs**: `runs/qwen3b_8bit/`
- **Training logs**: `logs/` (timestamped log files)
- **Final data**: `data/train.full.jsonl`, `data/eval.full.jsonl`

## Development Notes
- Training automatically resumes from last checkpoint if available
- TensorBoard logs every step for real-time monitoring
- Early stopping prevents overfitting
- Models save both adapters and tokenizer for easy loading
- Use `max_steps=10` in training config for quick dry-run testing