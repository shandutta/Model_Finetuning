from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

print("Loading dataset...")
ds = load_dataset('json', data_files='data/train.full.jsonl', split='train')

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-3B-Instruct')

print("Calculating token lengths...")
lengths = []
for e in ds:
    text = tok.apply_chat_template(e['messages'], tokenize=False)
    tokens = tok.encode(text)
    lengths.append(len(tokens))

print(f"\nResults:")
print(f"Max: {max(lengths)}")
print(f"P95: {np.percentile(lengths, 95):.0f}")
print(f"P90: {np.percentile(lengths, 90):.0f}")
print(f"Mean: {np.mean(lengths):.0f}")
print(f"Median: {np.median(lengths):.0f}")
print(f"Min: {min(lengths)}")