# Model Fine-Tuning Pipeline

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GPU: RTX 2080 Ti+](https://img.shields.io/badge/GPU-RTX%202080%20Ti+-orange)](https://developer.nvidia.com/cuda-gpus)

**Fine-tune coding language models on domain-specific data using QLoRA**

This repository provides a complete pipeline for fine-tuning the Qwen2.5-Coder-3B-Instruct model on custom datasets. Train models that understand your specific codebase, coding patterns, and domain knowledge using memory-efficient QLoRA on accessible GPU hardware.

## Features

- **Memory-efficient training** - QLoRA 8-bit quantization requires only 11GB VRAM
- **Multi-source data processing** - Combine conversation logs, git history, and code completion examples
- **Production-ready tools** - Comprehensive monitoring, evaluation, and comparison utilities
- **Automated workflows** - One-command training with integrated monitoring via tmux

## Getting Started

### Prerequisites

- **Hardware**: NVIDIA RTX 2080 Ti or better (11GB+ VRAM)
- **Software**: Python 3.12+, CUDA 11.8+
- **Storage**: 50GB+ available space
- **Memory**: 16GB+ RAM recommended

### Installation

```bash
git clone https://github.com/your-username/Model_Finetuning.git
cd Model_Finetuning
uv sync && source .venv/bin/activate

# Verify installation
# (No verification script available)
```

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

### Quick Start

```bash
# 1. Prepare training data
python scripts/data_prep/claude_logs/claude2sft.py --src data/claude_logs/ --out data/train.clean.jsonl
python scripts/data_prep/actual_code/repo2fim.py --repo /path/to/repo --out data/train.fim.jsonl
python scripts/data_prep/merge_and_split.py data/*.jsonl --name full --eval_frac 0.15

# 2. Start training with monitoring
./scripts.sh tmux-train

# 3. Evaluate results
python scripts/testing/compare_baseline_vs_lora.py \
  --prompt "How do I fix a KeyError in Python?" \
  --ckpt outputs/qwen3b_lora_8bit/checkpoint-650
```

For detailed workflows and options, see [scripts documentation](#scripts-reference).

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | Qwen/Qwen2.5-Coder-3B-Instruct | Pre-trained coding language model |
| Method | 8-bit QLoRA | Memory-efficient fine-tuning |
| LoRA Config | r=32, α=64, dropout=0.05 | Low-rank adaptation parameters |
| Learning Rate | 2e-5 (cosine schedule) | Training rate with warm-up |
| Batch Size | 24 (gradient accumulation) | Effective batch size |
| Max Length | 2048 tokens | Maximum sequence length |

## Monitoring

Training progress is monitored via TensorBoard and tmux sessions:

```bash
# Start training with integrated monitoring
./scripts.sh tmux-train

# Manual monitoring commands
./scripts.sh gpu                    # GPU utilization
./scripts.sh logs                   # Training logs
tensorboard --logdir runs/qwen3b_8bit --port 6006
```

## Use Cases

- **Code completion** - Generate contextually relevant code suggestions
- **Debugging assistance** - Provide domain-specific error analysis and fixes  
- **Code documentation** - Generate documentation matching project style
- **Codebase Q&A** - Answer questions about specific codebases and patterns

## Scripts Reference

### Data Processing
- `scripts/data_prep/claude_logs/claude2sft.py` - Convert conversation logs to training format
- `scripts/data_prep/actual_code/repo2fim.py` - Generate fill-in-the-middle examples from code
- `scripts/data_prep/merge_and_split.py` - Combine datasets and create train/eval splits
- `scripts/data_prep/qc_and_dedupe.py` - Quality control and deduplication

### Training
- `scripts/training/sft_qlora_8bit.py` - Production 8-bit QLoRA training
- `scripts/training/sft_qlora_4bit.py` - Development 4-bit QLoRA training

### Evaluation
- `scripts/testing/baseline.py` - Test baseline model performance
- `scripts/testing/compare_baseline_vs_lora.py` - Compare baseline vs fine-tuned models
- `scripts/testing/check_lengths.py` - Analyze dataset token length distribution

### Utilities
- `scripts.sh` - Convenience wrapper with tmux session management
- Run `./scripts.sh help` for all available commands

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen) for the base Qwen2.5-Coder model
- [Hugging Face](https://huggingface.co/) for transformers, datasets, and PEFT libraries
- [QLoRA authors](https://arxiv.org/abs/2305.14314) for memory-efficient fine-tuning methodology
