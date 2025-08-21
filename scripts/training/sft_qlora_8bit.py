# tmux new-session -d -s train "mkdir -p logs; python scripts/training/sft_qlora_8bit.py 2>&1 | tee -a logs/run_$(date +%F_%H-%M).log"
# tmux new-window -t train: -n run "python scripts/training/sft_qlora_8bit.py 2>&1 | tee -a logs/run_$(date +%F_%H-%M).log"

# Adjust these primarily
'''
num_train_epochs=5,
learning_rate=2e-5,                 # cooler than 1e-4
warmup_ratio=0.10,
lr_scheduler_type="cosine",
weight_decay=0.05,
max_grad_norm=1.0,
gradient_accumulation_steps=24,     # effective batch = 24 (1 x 24)

eval_steps=25,
save_steps=25, 
save_total_limit=3,

early_stopping_patience=5,       # stop after 5 evals with no meaningful improvement
early_stopping_threshold=0.002   # “meaningful improvement” = loss drops by ≥ 0.002
'''

# python sft_qlora_8bit.py
import os, torch, gc  
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import random, numpy as np

# Determinism
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

mid = "Qwen/Qwen2.5-Coder-3B-Instruct"
tok = AutoTokenizer.from_pretrained(mid, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"  # safer with packing=False

# --- 8-bit quant ---
bnb = BitsAndBytesConfig(
    load_in_8bit=True,
    # optional: llm_int8_threshold=6.0, llm_int8_skip_modules=None
)

model = AutoModelForCausalLM.from_pretrained(
    mid,
    quantization_config=bnb,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# --- data ---
ds = load_dataset("json", data_files="data/train.jsonl", split="train")
def fmt(e):
    if e.get("text"):                  # prefer prebuilt/clipped text
        return {"text": e["text"]}
    e["text"] = tok.apply_chat_template(
        e["messages"], tokenize=False, add_generation_prompt=False
    )
    return e
ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "text"])

eval_ds = None
try:
    eval_raw = load_dataset("json", data_files="data/eval.jsonl", split="train")
    eval_ds  = eval_raw.map(fmt, remove_columns=[c for c in eval_raw.column_names if c != "text"])
except Exception as e:
    print(f"[warn] no eval set: {e}")

# --- LoRA (bigger; you said VRAM is fine) ---
lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

# --- training config (slower & steadier) ---
args = SFTConfig(
    output_dir="outputs/qwen3b_lora_8bit",
    report_to=["tensorboard"],
    logging_dir=f"runs/qwen3b_8bit",
    logging_steps=1, 
    logging_first_step=True,
    disable_tqdm=False,

    per_device_eval_batch_size=1,
    eval_accumulation_steps=4,         

    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    dataloader_pin_memory=True,
    per_device_train_batch_size=1,
    dataset_text_field="text",
    max_length=2048,                    
    # max_steps=10,  # <- uncomment for a dry-run sanity check

    num_train_epochs=20,
    learning_rate=1e-5,                 # cooler than 1e-4
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    weight_decay=0.05,
    max_grad_norm=1.0,
    gradient_accumulation_steps=24,     # effective batch = 24 (1 x 24) - restored from 12     

    fp16=True,
    optim="adamw_bnb_8bit",

    gradient_checkpointing=True,        # match model.enable(...)
    packing=False,
    group_by_length=True,

    eval_strategy="steps" if eval_ds is not None else "no",
    save_strategy="steps", 

    eval_steps=10,
    save_steps=10, 
    save_total_limit=5,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
    seed=42, 
    greater_is_better=False,  # lower loss is better
)

class MemoryCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        """Force garbage collection after evaluation to prevent memory accumulation"""
        try:
            gc.collect()
            torch.cuda.empty_cache()
            # Print memory usage for monitoring
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                cached = torch.cuda.memory_reserved() / 1024**3     # GB
                print(f"[Memory] Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        except Exception as e:
            print(f"[Memory] Cleanup warning: {e}")
        return control

memory_callback = MemoryCallback()

trainer = SFTTrainer(
    model=model,
    args=args,
    peft_config=lora,
    train_dataset=ds,
    eval_dataset=eval_ds,
    processing_class=tok,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=6,       # stop after 6 evals with no meaningful improvement
            early_stopping_threshold=0.001   # "meaningful improvement" = loss drops by ≥ 0.001
        ),
        memory_callback  
    ]
)

if __name__ == "__main__":
    last = get_last_checkpoint(args.output_dir)
    if last:
        print(f"[resume] Resuming from {last}")
    else:
        print("[resume] No checkpoint found; training from scratch.")
    trainer.train(resume_from_checkpoint=last)    
    trainer.model.save_pretrained("outputs/qwen3b_lora_8bit/adapter", safe_serialization=True)
    tok.save_pretrained("outputs/qwen3b_lora_8bit/tokenizer")
    print("Saved adapters to outputs/qwen3b_lora_8bit/adapter")
