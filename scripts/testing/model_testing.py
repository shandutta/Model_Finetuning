#!/usr/bin/env python3
"""
Model Testing Script for RTX 2080 Ti (11GB VRAM)
Tests Qwen2.5-Coder models with different quantization strategies
"""

import argparse
import torch
import gc
import time
import threading
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.import_utils import is_flash_attn_2_available
from typing import Optional, Dict, Any

def print_memory_info(stage=""):
    """Print current memory usage with detailed breakdown"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_percent = (allocated / total_gpu) * 100
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB ({gpu_percent:.1f}%), Reserved: {reserved:.2f}GB")
    
    memory = psutil.virtual_memory()
    cpu_used = memory.used / 1e9
    cpu_total = memory.total / 1e9
    cpu_percent = memory.percent
    print(f"[{stage}] CPU Memory - Used: {cpu_used:.2f}GB / {cpu_total:.2f}GB ({cpu_percent:.1f}%)")
    print("-" * 70)

def monitor_cpu_memory_thread(stop_event, peak_memory):
    """Monitor peak CPU memory usage in background thread"""
    while not stop_event.is_set():
        current_usage = psutil.virtual_memory().used / 1e9
        peak_memory[0] = max(peak_memory[0], current_usage)
        time.sleep(0.1)

def measure_tokens_per_second(tokenizer, model, prompt, max_tokens: int = 200) -> Dict[str, Any]:
    """Measure token generation speed with accurate CUDA timing and peak mem.

    Gracefully handles CUDA OOM during generation by progressively reducing
    max tokens and optionally disabling cache.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warm up (best-effort; ignore OOM and continue with reduced settings)
    try:
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    except torch.cuda.OutOfMemoryError:
        clear_memory()

    # Track GPU peak memory during measurement
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Adaptive generation with OOM fallback
    attempts = 0
    cur_max_tokens = max_tokens
    use_cache = True
    outputs = None
    gen_err = None

    while attempts < 3 and outputs is None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cur_max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=use_cache,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as e:
            gen_err = e
            clear_memory()
            attempts += 1
            if attempts == 1:
                # First fallback: reduce tokens by half
                cur_max_tokens = max(32, cur_max_tokens // 2)
                print(f"[gen] CUDA OOM; retrying with {cur_max_tokens} tokens")
            elif attempts == 2:
                # Second fallback: disable KV cache
                use_cache = False
                print(f"[gen] Still OOM; retrying with use_cache=False")
            else:
                break

    if outputs is None:
        # Generation failed despite fallbacks
        raise gen_err if gen_err else RuntimeError("Generation failed")

    generation_time = time.time() - start_time
    new_tokens = int(outputs.shape[1] - inputs.input_ids.shape[1])
    tokens_per_second = new_tokens / generation_time if generation_time > 0 else 0.0

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    peak_gpu_alloc = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    peak_gpu_reserved = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0

    return {
        "tokens_per_second": tokens_per_second,
        "generation_time": generation_time,
        "new_tokens": new_tokens,
        "generated_text": generated_text,
        "peak_gpu_alloc": peak_gpu_alloc,
        "peak_gpu_reserved": peak_gpu_reserved,
    }

def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def _can_use_flash_attention_on_this_gpu() -> bool:
    """FlashAttention-2 generally requires SM80+ (Ampere). 2080 Ti is SM75."""
    if not torch.cuda.is_available():
        return False
    if not is_flash_attn_2_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    # Require SM80+ for reliability
    return (major * 10 + minor) >= 80


def test_model(
    model_name: str,
    quantization_config: BitsAndBytesConfig,
    max_memory: Optional[Dict[str, str]] = None,
    test_prompt: Optional[str] = None,
    flash_attention: bool = False,
    test_long_generation: bool = False,
    short_tokens: int = 100,
    long_tokens: int = 256,
):
    """Test a model with given configuration"""
    print(f"\n🧪 Testing: {model_name}")
    print(f"Quantization: {quantization_config}")
    if max_memory:
        print(f"Memory limits: {max_memory}")
    if flash_attention:
        print("FlashAttention: Requested")
    print("=" * 80)
    
    try:
        start_time = time.time()

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print_memory_info("After tokenizer")

        # Start CPU memory monitoring
        peak_cpu_memory = [psutil.virtual_memory().used / 1e9]
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(target=monitor_cpu_memory_thread, args=(stop_monitoring, peak_cpu_memory))
        monitor_thread.start()

        # Load model with adaptive OOM fallbacks
        print("Loading model...")
        load_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if max_memory:
            load_kwargs["max_memory"] = max_memory

        if flash_attention:
            # Guard FlashAttention for unsupported GPUs (e.g., RTX 2080 Ti SM75)
            if _can_use_flash_attention_on_this_gpu():
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using FlashAttention-2")
            else:
                print("[warn] FlashAttention-2 not supported on this GPU; falling back to eager")
                load_kwargs["attn_implementation"] = "eager"

        def _auto_offload_map() -> Dict[str, str]:
            avail_cpu_gb = int(psutil.virtual_memory().available / 1e9)
            cpu_budget = max(4, avail_cpu_gb - 6)
            # Conservative GPU budget for 2080 Ti to avoid spillover
            gpu_budget = "8GB"
            return {"0": gpu_budget, "cpu": f"{min(32, cpu_budget)}GB"}

        def _load_with(kwargs):
            clear_memory()
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        tried_steps = []
        model = None
        try:
            tried_steps.append("initial")
            model = _load_with(load_kwargs)
        except torch.cuda.OutOfMemoryError:
            print("[load] CUDA OOM on initial load. Attempting CPU offload...")
            # Retry with offloading if not already set
            if "max_memory" not in load_kwargs:
                load_kwargs["max_memory"] = _auto_offload_map()
            tried_steps.append("offload")
            try:
                model = _load_with(load_kwargs)
            except torch.cuda.OutOfMemoryError:
                # If 8-bit, fallback to 4-bit
                if getattr(quantization_config, "load_in_8bit", False):
                    print("[load] Still OOM. Falling back to 4-bit quantization...")
                    load_kwargs.pop("quantization_config", None)
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    tried_steps.append("4bit")
                    try:
                        model = _load_with(load_kwargs)
                    except torch.cuda.OutOfMemoryError as e:
                        print("[load] OOM even with 4-bit + offload.")
                        raise e
                else:
                    # Already in 4-bit; give up
                    raise

        model.config.use_cache = True

        load_time = time.time() - start_time
        step_note = " with offload" if "offload" in tried_steps else ""
        if "4bit" in tried_steps:
            print(f"✅ Model loaded in {load_time:.2f}s using 4-bit{step_note}")
        else:
            print(f"✅ Model loaded successfully in {load_time:.2f}s{step_note}")
        
        # Check memory footprint
        model_memory = model.get_memory_footprint() / 1e9
        print(f"Model memory footprint: {model_memory:.2f}GB")
        print_memory_info("After model load")
        
        # Test inference
        performance_results = {}
        if test_prompt:
            print("\n🚀 Testing inference...")
            
            # Short generation test
            print(f"Testing short generation ({short_tokens} tokens)...")
            short_result = measure_tokens_per_second(tokenizer, model, test_prompt, max_tokens=short_tokens)
            performance_results["short"] = short_result
            print(f"Short generation: {short_result['tokens_per_second']:.2f} tokens/sec")
            print_memory_info("After short generation")
            if torch.cuda.is_available():
                print(
                    f"[short] Peak GPU: alloc={short_result['peak_gpu_alloc']:.2f}GB, reserved={short_result['peak_gpu_reserved']:.2f}GB"
                )
            
            # Long generation test (if enabled)
            if test_long_generation:
                print(f"\nTesting long generation ({long_tokens} tokens)...")
                long_result = measure_tokens_per_second(tokenizer, model, test_prompt, max_tokens=long_tokens)
                performance_results["long"] = long_result
                print(f"Long generation: {long_result['tokens_per_second']:.2f} tokens/sec")
                print_memory_info("After long generation")
                if torch.cuda.is_available():
                    print(
                        f"[long] Peak GPU: alloc={long_result['peak_gpu_alloc']:.2f}GB, reserved={long_result['peak_gpu_reserved']:.2f}GB"
                    )
            
            print(f"\nGenerated text preview:\n{short_result['generated_text'][:200]}...")
        
        # Stop monitoring and get peak CPU usage
        stop_monitoring.set()
        monitor_thread.join()
        peak_cpu_gb = peak_cpu_memory[0]
        
        # Cleanup
        del model, tokenizer
        clear_memory()
        
        print(f"✅ Model test completed successfully!")
        print(f"Peak CPU Memory: {peak_cpu_gb:.2f}GB")
        if performance_results:
            print(f"Performance Summary:")
            for test_type, result in performance_results.items():
                print(f"  {test_type}: {result['tokens_per_second']:.2f} tok/s ({result['generation_time']:.2f}s)")
        print()
        
        return {
            "success": True,
            "load_time": load_time,
            "model_memory": model_memory,
            "peak_cpu_memory": peak_cpu_gb,
            "performance": performance_results
        }
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA Out of Memory during {model_name} load or generation.")
        print("   Tried: initial → offload → 4-bit (if applicable)")
        print("💡 Tip: Reduce --gpu-mem budget or skip 8-bit on this GPU")
        clear_memory()
        return {"success": False, "error": "CUDA OOM"}
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        clear_memory()
        return {"success": False, "error": str(e)}

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Benchmark Qwen2.5-Coder inference on RTX 2080 Ti")
    parser.add_argument("--models", nargs="+", default=["7b", "14b"], choices=["7b", "14b", "32b"], help="Model sizes to test")
    parser.add_argument("--quant", default="auto", choices=["auto", "4bit", "8bit", "both"], help="Quantization to include")
    parser.add_argument("--short-tokens", type=int, default=100, help="Short generation token count")
    parser.add_argument("--long-tokens", type=int, default=256, help="Long generation token count; 0 to skip")
    parser.add_argument("--no-long", action="store_true", help="Disable long generation tests")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt string to test")
    parser.add_argument("--prompt-file", type=str, default=None, help="File with prompt text")
    parser.add_argument("--fa2", action="store_true", help="Request FlashAttention-2 if supported")
    parser.add_argument("--gpu-mem", type=str, default=None, help="Override GPU mem budget for offload (e.g., 9GB)")
    parser.add_argument("--cpu-budget-gb", type=int, default=None, help="Override CPU mem budget for offload (GB)")
    parser.add_argument("--skip-large-if-low-ram", action="store_true", help="Skip 14B/32B if low free RAM")
    args = parser.parse_args()

    print("🔧 Model Testing Script (RTX 2080 Ti focus)")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.2f}GB | CC: {major}.{minor}")
    else:
        print("GPU: None detected")
    print_memory_info("Initial")

    # Test prompt for code generation
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            test_prompt = f.read()
    else:
        test_prompt = args.prompt or (
            """def fibonacci(n):
    \"\"\"Generate fibonacci sequence up to n terms\"\"\"
    """
        )
    
    # Test configurations including offloading strategies
    # Each config tests different memory management approaches:
    # - No offloading: Model runs entirely on GPU (fastest)
    # - Moderate offload: Some layers on CPU (balanced capability/speed)
    # - Aggressive offload: More layers on CPU (higher capability, slower)
    # - Extreme offload: Most layers on CPU (maximum capability, slowest)
    # Determine available CPU memory headroom for offloading
    avail_cpu_gb = int(psutil.virtual_memory().available / 1e9)
    # Leave a safety buffer for the OS and other processes
    auto_cpu_budget_gb = max(4, avail_cpu_gb - 6)
    cpu_budget_gb = args.cpu_budget_gb if args.cpu_budget_gb is not None else auto_cpu_budget_gb

    test_configs = [
        {
            "name": "Qwen2.5-Coder-7B-Instruct (8-bit)",
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "quantization": BitsAndBytesConfig(load_in_8bit=True),
            "max_memory": None,
            "flash_attention": args.fa2,
            "test_long": (args.long_tokens > 0) and (not args.no_long)
        },
        {
            "name": "Qwen2.5-Coder-7B-Instruct (4-bit)",
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "quantization": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            "max_memory": None,
            "flash_attention": args.fa2,
            "test_long": (args.long_tokens > 0) and (not args.no_long)
        },
        {
            "name": "Qwen2.5-Coder-14B-Instruct (4-bit + moderate offload)",
            "model": "Qwen/Qwen2.5-Coder-14B-Instruct", 
            "quantization": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            "max_memory": {"0": (args.gpu_mem or "9GB"), "cpu": f"{min(24, cpu_budget_gb)}GB"},
            "flash_attention": args.fa2,
            "test_long": (args.long_tokens > 0) and (not args.no_long)
        },
        {
            "name": "Qwen2.5-Coder-14B-Instruct (4-bit + aggressive offload)",
            "model": "Qwen/Qwen2.5-Coder-14B-Instruct", 
            "quantization": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            "max_memory": {"0": (args.gpu_mem or "7GB"), "cpu": f"{min(32, cpu_budget_gb)}GB"},
            "flash_attention": args.fa2,
            "test_long": (args.long_tokens > 0) and (not args.no_long)
        },
        # FlashAttention is not supported on 2080 Ti (SM75); skip by default
    ]

    # Filter by --models selection first, then add 32B if needed
    filtered = []
    for cfg in test_configs:
        if ("7B" in cfg["name"] and "7b" not in args.models):
            continue
        if ("14B" in cfg["name"] and "14b" not in args.models):
            continue
        filtered.append(cfg)
    
    # Add 32B extreme offload when requested (after initial filtering)
    if "32b" in args.models and args.quant in ("4bit", "both", "auto"):
        filtered.append({
            "name": "Qwen2.5-Coder-32B-Instruct (4-bit + extreme offload)",
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "quantization": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            "max_memory": {"0": (args.gpu_mem or "8GB"), "cpu": f"{min(64, cpu_budget_gb)}GB"},
            "flash_attention": args.fa2,
            "test_long": (args.long_tokens > 0) and (not args.no_long)
        })

    # Filter by --quant selection
    test_configs = []
    for cfg in filtered:
        if args.quant == "4bit" and "(8-bit)" in cfg["name"]:
            continue
        if args.quant == "8bit" and "(4-bit)" in cfg["name"]:
            continue
        test_configs.append(cfg)

    # Optionally skip large configs if RAM is constrained
    results = {}
    for config in test_configs:
        # More nuanced RAM checking: 14B needs 12GB+, 32B needs 20GB+
        if args.skip_large_if_low_ram:
            if "32B" in config["name"] and cpu_budget_gb < 20:
                print(f"[skip] Skipping {config['name']} due to insufficient CPU RAM (~{cpu_budget_gb}GB < 20GB required)")
                continue
            elif "14B" in config["name"] and cpu_budget_gb < 12:
                print(f"[skip] Skipping {config['name']} due to insufficient CPU RAM (~{cpu_budget_gb}GB < 12GB required)")
                continue
        result = test_model(
            model_name=config["model"],
            quantization_config=config["quantization"],
            max_memory=config["max_memory"],
            test_prompt=test_prompt,
            flash_attention=config["flash_attention"],
            test_long_generation=config["test_long"],
            short_tokens=args.short_tokens,
            long_tokens=max(0, args.long_tokens),
        )
        results[config["name"]] = result
        
        # Wait between tests for memory cleanup (longer for large models)
        cleanup_time = 8 if any(x in config["name"] for x in ["14B", "32B"]) else 4
        print(f"⏳ Waiting {cleanup_time}s for memory cleanup...")
        time.sleep(cleanup_time)
    
    # Summary
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*80)
    
    successful_configs = []
    performance_comparison = []
    
    for name, result in results.items():
        if result["success"]:
            status = "✅ SUCCESS"
            successful_configs.append(name)
            
            # Extract performance data
            perf_data = {
                "name": name,
                "load_time": result.get("load_time", 0),
                "model_memory": result.get("model_memory", 0),
                "peak_cpu_memory": result.get("peak_cpu_memory", 0),
                "short_tokens_per_sec": result.get("performance", {}).get("short", {}).get("tokens_per_second", 0),
                "long_tokens_per_sec": result.get("performance", {}).get("long", {}).get("tokens_per_second", 0)
            }
            performance_comparison.append(perf_data)
            
            print(f"\n{name}: {status}")
            print(f"  Load time: {result.get('load_time', 0):.1f}s")
            print(f"  GPU memory: {result.get('model_memory', 0):.2f}GB")
            print(f"  Peak CPU: {result.get('peak_cpu_memory', 0):.2f}GB")
            
            if result.get("performance"):
                perf = result["performance"]
                if "short" in perf:
                    print(f"  Short gen: {perf['short']['tokens_per_second']:.2f} tok/s")
                if "long" in perf:
                    print(f"  Long gen: {perf['long']['tokens_per_second']:.2f} tok/s")
        else:
            status = "❌ FAILED"
            print(f"\n{name}: {status}")
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Performance ranking
    if performance_comparison:
        print("\n🏆 PERFORMANCE RANKING (by tokens/second):")
        performance_comparison.sort(key=lambda x: x["short_tokens_per_sec"], reverse=True)
        for i, config in enumerate(performance_comparison[:3], 1):
            speed_indicator = "🚀" if i == 1 else "⚡" if i == 2 else "🔥"
            print(f"  {i}. {speed_indicator} {config['name']}: {config['short_tokens_per_sec']:.2f} tok/s")
    
    # Memory efficiency ranking
    if performance_comparison:
        print("\n💾 MEMORY EFFICIENCY RANKING:")
        memory_configs = [c for c in performance_comparison if c["model_memory"] > 0]
        memory_configs.sort(key=lambda x: x["model_memory"])
        for i, config in enumerate(memory_configs[:3], 1):
            memory_indicator = "🟢" if i == 1 else "🟡" if i == 2 else "🟠"
            cpu_extra = max(0, config["peak_cpu_memory"] - config["model_memory"])
            print(f"  {i}. {memory_indicator} {config['name']}: {config['model_memory']:.2f}GB GPU + {cpu_extra:.2f}GB CPU")
    
    # Offloading impact analysis
    offload_configs = [c for c in performance_comparison if "offload" in c["name"]]
    if offload_configs:
        print("\n🔄 OFFLOADING IMPACT ANALYSIS:")
        # Get baseline CPU usage from non-offload config
        baseline_cpu = 0
        baseline_config = next((c for c in performance_comparison if "7B-Instruct (4-bit)" in c["name"] and "offload" not in c["name"]), None)
        if baseline_config:
            baseline_cpu = baseline_config["peak_cpu_memory"]
        
        for config in offload_configs:
            # More accurate CPU offload calculation
            cpu_offload = max(0, config["peak_cpu_memory"] - baseline_cpu)
            speed_penalty = ""
            if (config["short_tokens_per_sec"] > 0 and baseline_config and 
                baseline_config["short_tokens_per_sec"] > 0 and 
                baseline_config["short_tokens_per_sec"] != config["short_tokens_per_sec"]):
                penalty = ((baseline_config["short_tokens_per_sec"] - config["short_tokens_per_sec"]) / baseline_config["short_tokens_per_sec"]) * 100
                speed_penalty = f" ({penalty:.0f}% slower)" if penalty > 0 else f" ({abs(penalty):.0f}% faster)"
            
            print(f"  • {config['name']}: {cpu_offload:.2f}GB CPU offload{speed_penalty}")
    
    # Recommendations
    print("\n🎯 SMART RECOMMENDATIONS:")
    
    # Find best overall option
    best_config = None
    if performance_comparison:
        # Score based on: speed * 0.4 + (1/memory_usage) * 0.3 + success_bonus * 0.3
        for config in performance_comparison:
            speed_score = config["short_tokens_per_sec"] / 10  # Normalize
            memory_score = 10 / max(config["model_memory"], 1)  # Inverse of memory usage
            total_score = speed_score * 0.4 + memory_score * 0.3 + 3.0  # Base success bonus
            config["score"] = total_score
        
        best_config = max(performance_comparison, key=lambda x: x["score"])
    
    if best_config:
        print(f"🥇 BEST OVERALL: {best_config['name']}")
        print(f"   • {best_config['short_tokens_per_sec']:.2f} tokens/sec")
        print(f"   • {best_config['model_memory']:.2f}GB GPU memory")
        print(f"   • {best_config['peak_cpu_memory']:.2f}GB peak CPU memory")
        
        if "32B" in best_config['name']:
            print("   💡 Amazing! You can run 32B models with offloading!")
        elif "14B" in best_config['name']:
            print("   🎉 Great! 14B models work well with your setup!")
        elif "7B" in best_config['name']:
            print("   ✨ Solid choice! 7B models run smoothly on your RTX 2080 Ti!")
    
    # Specific recommendations
    print("\n📋 SPECIFIC RECOMMENDATIONS:")
    if any("32B" in name for name in successful_configs):
        print("• You can run 32B models! Use aggressive CPU offloading for maximum capabilities")
    elif any("14B" in name for name in successful_configs):
        print("• 14B models work great! Use moderate offloading for best balance")
    elif any("7B" in name for name in successful_configs):
        print("• Stick with 7B models for reliable performance on your hardware")
    else:
        print("• Consider staying with your current 3B model or upgrading hardware")
    
    print("\n💡 OPTIMIZATION TIPS:")
    print("• Models with CPU offloading trade speed for capability")
    print("• FlashAttention can reduce memory usage by ~10-15%")
    print("• 4-bit quantization offers the best memory/quality balance")
    print("• Consider batch size = 1 for maximum model size capability")

if __name__ == "__main__":
    # Check requirements
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires a GPU.")
        exit(1)
    
    if torch.cuda.get_device_properties(0).total_memory < 10e9:
        print("⚠️  Warning: GPU has less than 10GB memory. Tests may fail.")
    
    main()
