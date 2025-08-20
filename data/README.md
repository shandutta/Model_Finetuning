# Data Directory

This directory contains training datasets for the QLoRA fine-tuning pipeline. All data files are excluded from Git to keep the repository lightweight.

## 📂 Directory Structure

```
data/
├── claude_logs/                   # Claude conversation logs → SFT format
│   ├── train.clean.jsonl         # Processed Claude conversations
│   └── train.follow3m.clean.jsonl # Follow-up conversations
├── git_history/                   # Git commits → SFT examples
│   ├── git_sft.clean.jsonl       # Processed git commit data
│   └── git_sft.messages.jsonl    # Git history in message format
├── actual_code/                   # Real codebases → FIM examples
│   ├── train.fim.clean.jsonl     # Fill-in-the-middle examples
│   └── train.fim.jsonl           # Raw FIM data
└── staging/                       # Final datasets for training
    ├── train.full.jsonl          # Combined training data
    └── eval.full.jsonl           # Evaluation split
```

## 🔄 Data Processing Pipeline

### 1. Source Data Collection
- **Claude Logs**: Export conversation logs from Claude sessions
- **Git Repositories**: Clone or access codebases for commit history
- **Code Files**: Real Python/JavaScript/etc. files for FIM generation

### 2. Data Preparation Scripts
```bash
# Convert Claude logs to supervised fine-tuning format
python scripts/data_prep/claude_logs/claude2sft.py --src data/claude_logs/ --out data/train.clean.jsonl

# Extract git commit history and convert to SFT format
python scripts/data_prep/actual_code/git_to_sft.py --repo /path/to/repo --out data/git_history/git_sft.jsonl

# Generate Fill-in-the-Middle examples from codebases
python scripts/data_prep/actual_code/repo2fim.py --repo /path/to/repo --out data/actual_code/train.fim.jsonl --per_file 6 --max_len 1600

# Merge all datasets and create train/eval split
python scripts/data_prep/merge_and_split.py \
  data/claude_logs/train.clean.jsonl \
  data/git_history/git_sft.clean.jsonl \
  data/actual_code/train.fim.clean.jsonl \
  --name full --eval_frac 0.15
```

## 📊 Data Format

All training data uses the chat format expected by Qwen models:

```json
{
  "messages": [
    {
      "role": "system", 
      "content": "You are a precise coding assistant that diagnoses errors step-by-step, references file paths/lines when useful, and proposes concrete fixes."
    },
    {
      "role": "user", 
      "content": "Python traceback:\nKeyError: 'user_id'\nHow do I diagnose and fix this?"
    },
    {
      "role": "assistant", 
      "content": "This KeyError indicates that your code is trying to access a dictionary key 'user_id' that doesn't exist..."
    }
  ]
}
```

## 🎯 Data Quality Guidelines

### Claude Logs
- Focus on debugging, error resolution, and code explanation conversations
- Remove personal information and sensitive data
- Filter for high-quality, educational exchanges

### Git History
- Include meaningful commit messages with code context
- Focus on bug fixes and feature implementations
- Filter out merge commits and trivial changes

### FIM Examples
- Generate from diverse, high-quality codebases
- Balance different programming languages and patterns
- Ensure proper syntax and semantic correctness

## 📈 Dataset Statistics

| Dataset | Examples | Avg Length | Source |
|---------|----------|------------|--------|
| Claude Logs | ~2,500 | 850 tokens | Conversation exports |
| Git History | ~1,200 | 650 tokens | Repository commits |
| FIM Examples | ~3,800 | 400 tokens | Code repositories |
| **Total** | **~7,500** | **630 tokens** | Multi-source |

## 🚨 Important Notes

- **Privacy**: Ensure all data is properly anonymized before processing
- **Licensing**: Verify that source code repositories have appropriate licenses
- **Quality**: Review generated examples for accuracy and relevance
- **Size**: Monitor total dataset size to ensure efficient training

## 🔗 Related Scripts

- `scripts/data_prep/qc_and_dedupe.py` - Quality control and deduplication
- `scripts/data_prep/len_scan.py` - Token length analysis
- `scripts/data_prep/merge_and_split.py` - Dataset merging and splitting
- `scripts/testing/check_lengths.py` - Analyze token distributions

---

*This directory structure supports the complete data pipeline from raw sources to training-ready datasets.*
