import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

mid = "Qwen/Qwen2.5-Coder-3B-Instruct"
tok = AutoTokenizer.from_pretrained(mid, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(mid, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = prepare_model_for_kbit_training(model)

ds = load_dataset("json", data_files="data/train.jsonl", split="train")
def fmt(e):
    e["text"] = tok.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)
    return e
ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "text"])

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

args = SFTConfig(
    output_dir="outputs/qwen3b_lora_8bit",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    optim="adamw_bnb_8bit",      # works with bitsandbytes
    logging_steps=5,
    save_total_limit=1,
    report_to=[],
    dataset_text_field="text",
    packing=False,
    max_length=1536,         # put length here (not on SFTTrainer)
)

trainer = SFTTrainer(
    model=model,
    args=args,
    peft_config=lora,
    train_dataset=ds,
    processing_class=tok,         # replaces `tokenizer=tok`
)

if __name__ == "__main__":
    trainer.train()
    trainer.model.save_pretrained("outputs/qwen3b_lora_8bit/adapter")
    tok.save_pretrained("outputs/qwen3b_lora_8bit/tokenizer")
    print("Saved adapters to outputs/qwen3b_lora_8bit/adapter")
