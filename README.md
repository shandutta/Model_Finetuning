# Model Fine-Tuning Pipeline

**QLoRA fine-tuning pipeline for coding assistants using Qwen2.5-Coder-3B-Instruct**

This project implements a complete pipeline for fine-tuning coding language models using QLoRA (Quantized Low-Rank Adaptation) with multi-source training data including Claude conversation logs, Git commit history, and Fill-in-the-Middle (FIM) examples from actual codebases.

## Project Goals

- **Improve code completion and debugging** capabilities of base coding models
- **Multi-source data integration** for comprehensive training coverage
- **Memory-efficient training** using 8-bit and 4-bit quantization with LoRA adapters
- **Reproducible research** with systematic evaluation and comparison tools

## Architecture

### Data Pipeline
```
Claude Logs → SFT Format
Git History → SFT Format  } → Merged Dataset → Train/Eval Split → QLoRA Training
Real Code → FIM Examples
```

### Model Training
- **Base Model**: `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Method**: QLoRA with 8-bit quantization (BitsAndBytesConfig)
- **LoRA Config**: r=32, alpha=64, dropout=0.05
- **Training**: Early stopping, gradient accumulation, cosine scheduling

## Project Structure

```
Model_Finetuning/
├── data/                           # Training datasets (gitignored)
│   ├── claude_logs/               # Processed Claude conversation logs
│   ├── git_history/               # Git commit → SFT examples
│   ├── actual_code/               # FIM examples from codebases
│   └── staging/                   # Final train/eval datasets
├── scripts/
│   ├── data_prep/                 # Data processing pipeline
│   │   ├── claude_logs/          # Claude log → SFT conversion
│   │   └── actual_code/          # Code → FIM generation
│   ├── training/                  # QLoRA training scripts
│   └── testing/                   # Model evaluation tools
├── outputs/                       # Model checkpoints (gitignored)
├── runs/                          # TensorBoard logs (gitignored)
└── logs/                          # Training logs (gitignored)
```

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### 2. Data Preparation
```bash
# Convert Claude logs to SFT format
python scripts/data_prep/claude_logs/claude2sft.py --src data/claude_logs/ --out data/train.clean.jsonl

# Generate FIM examples from codebase
python scripts/data_prep/actual_code/repo2fim.py --repo /path/to/repo --out data/train.fim.jsonl --per_file 6 --max_len 1600

# Merge datasets and create train/eval split
python scripts/data_prep/merge_and_split.py data/claude_logs/train.clean.jsonl data/actual_code/train.fim.jsonl --name full --eval_frac 0.15
```

### 3. Training
```bash
# Start training with tmux session monitoring
./scripts.sh tmux-train

# Or direct training
python scripts/training/sft_qlora_8bit.py
```

### 4. Evaluation
```bash
# Compare baseline vs fine-tuned model
python scripts/testing/compare_baseline_vs_lora.py \
  --prompt "How do I fix a KeyError in Python?" \
  --ckpt outputs/qwen3b_lora_8bit/checkpoint-650
```

## Key Features

### Data Sources
- **Claude Conversation Logs**: Real debugging and coding assistance conversations
- **Git Commit History**: Code changes with commit messages as context
- **Fill-in-the-Middle**: Code completion examples from actual repositories

### Training Optimizations
- **8-bit quantization** for memory efficiency (11GB VRAM compatible)
- **LoRA adapters** for parameter-efficient fine-tuning
- **Early stopping** with validation monitoring
- **Gradient accumulation** for effective batch size scaling

### Monitoring & Tools
- **TensorBoard integration** for real-time training metrics
- **Automated checkpointing** with best model selection
- **Comparison tools** for baseline vs fine-tuned evaluation
- **Tmux automation** for multi-window training monitoring

## Training Configuration

### Model Parameters
- **Base**: Qwen/Qwen2.5-Coder-3B-Instruct
- **Quantization**: 8-bit with BitsAndBytesConfig
- **LoRA**: r=32, alpha=64, dropout=0.05
- **Target modules**: All attention and MLP projections

### Training Hyperparameters
- **Learning rate**: 2e-5 with cosine scheduling
- **Batch size**: 1 per device, 24 gradient accumulation steps
- **Max length**: 2048 tokens
- **Epochs**: 20 with early stopping (patience=6, threshold=0.001)

## Hardware Requirements

- **GPU**: RTX 2080 Ti (11GB VRAM) or better
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for datasets and checkpoints

## Results & Evaluation

Training metrics and model comparisons are logged via TensorBoard:
```bash
tensorboard --logdir runs/qwen3b_8bit --port 6006
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with your data
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **Qwen Team** for the base Qwen2.5-Coder model
- **Hugging Face** for transformers, datasets, and PEFT libraries
- **QLoRA authors** for memory-efficient fine-tuning methodology

---

*This project demonstrates practical application of QLoRA fine-tuning for coding assistants with multi-source data integration and systematic evaluation.*