# python sft_qlora.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

mid = "Qwen/Qwen2.5-Coder-3B-Instruct"
tok = AutoTokenizer.from_pretrained(mid, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# QLoRA (4-bit) to free VRAM, then spend it on length/batch
bnb = BitsAndBytesConfig(
    load_in_8bit=True
    # optional knobs:
    # llm_int8_threshold=6.0,          # keep default unless you see overflow warnings
    # llm_int8_skip_modules=None,   
)

model = AutoModelForCausalLM.from_pretrained(
    mid,
    quantization_config=bnb,
    torch_dtype=torch.float16,  # good for 2080 Ti
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(                            # Recommended by PyTorch
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# Data
ds = load_dataset("json", data_files="data/train.jsonl", split="train")
def fmt(e):
    e["text"] = tok.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)
    return e
ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "text"])

# Bigger LoRA (still safe on 3B with 4-bit)
lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

# Heavier training config
args = SFTConfig(
    # visibility
    logging_steps=1,
    logging_first_step=True,
    report_to=["tensorboard"],                    # or ["tensorboard"] if you want TB
    disable_tqdm=False,
    logging_dir = "runs/qwen3b",

    output_dir="outputs/qwen3b_qlora_4bit",
    per_device_train_batch_size=2,         # try 3 first; bump to 4 if VRAM allows
    gradient_accumulation_steps=1,         # effective batch = 24
    num_train_epochs=2,                    # more passes over your data
    learning_rate=1e-4,                    # a bit gentler for larger LoRA
    fp16=True,
    optim="adamw_bnb_8bit",
    save_total_limit=2,
    dataset_text_field="text",
    packing=False,                         
    max_length=2048,                       # longer context to capture your chats
    gradient_checkpointing=False,          # cuts activation VRAM (slower = ok)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    group_by_length=True,                  # less padding → better throughput
    seed=42,
    dataloader_num_workers=0,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    peft_config=lora,
    train_dataset=ds,
    processing_class=tok,
)

if __name__ == "__main__":
    trainer.train()
    trainer.model.save_pretrained("outputs/qwen3b_qlora_4bit/adapter")
    tok.save_pretrained("outputs/qwen3b_qlora_4bit/tokenizer")
    print("Saved adapters to outputs/qwen3b_qlora_4bit/adapter")
