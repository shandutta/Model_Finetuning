#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for vLLM deployment.
This creates a unified model that vLLM can serve directly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import argparse
from pathlib import Path

def merge_lora_adapter(
    base_model_path="Qwen/Qwen2.5-Coder-3B-Instruct",
    adapter_path="outputs/qwen3b_lora_8bit/adapter",
    output_path="outputs/qwen3b_merged",
    load_in_8bit=False  # Don't use quantization for merging
):
    """
    Merge LoRA adapter with base model for vLLM serving
    
    Args:
        base_model_path: Path to base model
        adapter_path: Path to LoRA adapter
        output_path: Where to save merged model
        load_in_8bit: Whether to load base model in 8-bit (not recommended for merging)
    """
    
    print("🔄 Starting LoRA merge process...")
    print(f"📥 Base model: {base_model_path}")
    print(f"🔧 Adapter: {adapter_path}")
    print(f"💾 Output: {output_path}")
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        use_fast=True, 
        trust_remote_code=True
    )
    
    # Load base model
    print("📥 Loading base model...")
    if load_in_8bit:
        print("⚠️  Loading in 8-bit (may cause issues during merge)")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("✅ Loading in full precision (recommended for merging)")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Load LoRA adapter
    print("🔧 Loading LoRA adapter...")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge adapter into base model
    print("🔗 Merging LoRA adapter with base model...")
    merged_model = model_with_adapter.merge_and_unload()
    
    # Save merged model
    print(f"💾 Saving merged model to {output_path}...")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"  # Split large models into chunks
    )
    
    # Save tokenizer
    print("💾 Saving tokenizer...")
    tokenizer.save_pretrained(output_path)
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"🖥️  GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    print(f"✅ Merge completed! Merged model saved to: {output_path}")
    print(f"📁 Model files: {list(output_dir.glob('*'))}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-3B-Instruct", 
                       help="Base model path")
    parser.add_argument("--adapter", default="outputs/qwen3b_lora_8bit/adapter", 
                       help="LoRA adapter path")
    parser.add_argument("--output", default="outputs/qwen3b_merged", 
                       help="Output path for merged model")
    parser.add_argument("--load_in_8bit", action="store_true", 
                       help="Load base model in 8-bit (not recommended)")
    
    args = parser.parse_args()
    
    # Check if adapter exists
    if not os.path.exists(args.adapter):
        print(f"❌ Error: Adapter not found at {args.adapter}")
        return
    
    # Perform merge
    try:
        output_path = merge_lora_adapter(
            base_model_path=args.base_model,
            adapter_path=args.adapter,
            output_path=args.output,
            load_in_8bit=args.load_in_8bit
        )
        
        print("\n" + "="*60)
        print("🎯 Next Steps:")
        print("="*60)
        print(f"1. Your merged model is ready at: {output_path}")
        print("2. You can now serve it with vLLM:")
        print(f"   vllm serve {output_path} --host 0.0.0.0 --port 8000")
        print("3. Or use the API server script we'll create next!")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        raise

if __name__ == "__main__":
    main()