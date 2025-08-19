from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
mid="Qwen/Qwen2.5-Coder-3B-Instruct"
tok=AutoTokenizer.from_pretrained(mid, use_fast=True, trust_remote_code=True)
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
m=AutoModelForCausalLM.from_pretrained(mid, quantization_config=bnb, device_map="auto", trust_remote_code=True)
pipe=pipeline("text-generation", model=m, tokenizer=tok)
prompt="You are a precise coding assistant.\nUser: Python traceback:\nKeyError: 'user_id'\nHow do I diagnose and fix this?\nAssistant:"
print(pipe(prompt, max_new_tokens=160, do_sample=False)[0]["generated_text"])
