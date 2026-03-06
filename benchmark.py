#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HF-LLM-Profiler Multi-Model Benchmark Script
Profile single Transformer Block for time/memory/FLOPs, then scale to estimate full model
Only download config.json, not full model weights
"""

import torch
import json
import os
import sys
import subprocess
import multiprocessing
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Profiler import ModelProfiler

# ============================================================
# Configuration
# ============================================================

# Device configuration
DEVICE = "cuda:5"

# Test configuration
CONFIG = {
    "batch_size": 8,
    "seq_len": 512,
    "skip_round": 5,
    "test_round": 10,
}

# LoRA configuration for fine-tuning tests
LORA_CONFIG = {
    "r": 8,                    # LoRA rank
    "lora_alpha": 16,          # LoRA alpha scaling
    "lora_dropout": 0.05,      # Dropout probability
    "bias": "none",            # Bias type
    "target_modules": ["q_proj", "v_proj"],  # Target attention modules
}

# HFD downloader path
HFD_SCRIPT = "/data/HUGGINGFACE/hfd.sh"

# Model save directory
MODEL_DIR = "/workspace/code/HF-LLM-Profiler/models"

# Output file
OUTPUT_FILE = "bench/measured_data.json"

# Hugging Face credentials (for gated models)
HF_USERNAME = "huluhuluu"
HF_TOKEN = "hf_abababababababababababababababab" # Replace with your actual token or set as environment variable for security

# ============================================================
# Test Model List
# Ranging from ~0.5B to 100B parameters
# ============================================================

MODELS = [
    # ============ Small Models (< 1B) ============
    {"id": "Qwen/Qwen2.5-0.5B", "name": "Qwen2.5-0.5B", "params_b": 0.5, "dtype": torch.bfloat16},
    {"id": "Qwen/Qwen2.5-1.5B", "name": "Qwen2.5-1.5B", "params_b": 1.5, "dtype": torch.bfloat16},
    {"id": "microsoft/phi-2", "name": "Phi-2", "params_b": 2.7, "dtype": torch.bfloat16},
    
    # ============ Medium Models (3B - 15B) ============
    {"id": "Qwen/Qwen2.5-3B", "name": "Qwen2.5-3B", "params_b": 3, "dtype": torch.bfloat16},
    {"id": "Qwen/Qwen2.5-7B", "name": "Qwen2.5-7B", "params_b": 7, "dtype": torch.bfloat16},
    {"id": "meta-llama/Llama-3.2-3B", "name": "Llama-3.2-3B", "params_b": 3, "dtype": torch.bfloat16},
    {"id": "meta-llama/Meta-Llama-3-8B", "name": "Llama-3-8B", "params_b": 8, "dtype": torch.bfloat16},
    {"id": "mistralai/Mistral-7B-v0.1", "name": "Mistral-7B", "params_b": 7, "dtype": torch.bfloat16},
    
    # ============ Large Models (15B - 70B) ============
    {"id": "Qwen/Qwen2.5-14B", "name": "Qwen2.5-14B", "params_b": 14, "dtype": torch.bfloat16},
    {"id": "Qwen/Qwen2.5-32B", "name": "Qwen2.5-32B", "params_b": 32, "dtype": torch.bfloat16},
    {"id": "meta-llama/Meta-Llama-3.1-70B", "name": "Llama-3.1-70B", "params_b": 70, "dtype": torch.bfloat16},
    
    # ============ Extra Large Models (> 70B) ============
    {"id": "Qwen/Qwen2.5-72B", "name": "Qwen2.5-72B", "params_b": 72, "dtype": torch.bfloat16},
    {"id": "deepseek-ai/DeepSeek-V3", "name": "DeepSeek-V3", "params_b": 37, "is_moe": True, "dtype": torch.bfloat16,
     "lora_target_modules": ["q_a_proj", "q_b_proj"]}
]

# ============================================================
# Utility Functions
# ============================================================

def download_config(model_id: str, model_dir: str) -> str:
    """
    Download model's config.json
    Returns local path
    """
    from huggingface_hub import login, hf_hub_download
    
    # Create model directory
    model_name = model_id.split('/')[-1]
    local_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(os.path.join(local_path, "config.json")):
        print(f"  config.json already exists: {local_path}")
        return local_path
    
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"  Downloading config.json: {model_id}")
    
    # Try hfd first
    cmd = [
        HFD_SCRIPT,
        model_id,
        "--include", "^config.json$",
        "--local-dir", local_path,
        "--hf_username", HF_USERNAME,
        "--hf_token", HF_TOKEN,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(os.path.join(local_path, "config.json")):
        return local_path
    
    # hfd failed, use huggingface_hub directly
    print(f"  hfd failed, trying huggingface_hub...")
    try:
        login(token=HF_TOKEN)
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_dir=local_path,
            token=HF_TOKEN,
        )
        print(f"  ✓ Download success: {config_path}")
        return local_path
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        raise


def parse_flops(flops_str):
    """Parse FLOPs string, return GFLOPs value"""
    if not flops_str or flops_str == "N/A":
        return 0.0
    try:
        parts = flops_str.split()
        value = float(parts[0])
        unit = parts[1] if len(parts) > 1 else ""
        
        if "TFLOP" in unit:
            return value * 1000
        elif "GFLOP" in unit:
            return value
        elif "MFLOP" in unit:
            return value / 1000
        return value
    except:
        return 0.0


def parse_params(params_str):
    """Parse params string, return B (billions)"""
    if not params_str or params_str == "N/A":
        return 0.0
    try:
        parts = params_str.split()
        value = float(parts[0])
        unit = parts[1] if len(parts) > 1 else ""
        
        if "B" in unit:
            return value
        elif "M" in unit:
            return value / 1000
        elif "K" in unit:
            return value / 1e6
        return value
    except:
        return 0.0


# ============================================================
# Test Functions (run in subprocess)
# ============================================================

def run_flops_test(model_path, bs, seq_len, device, dtype, result_queue):
    """Test FLOPs"""
    try:
        import torch
        profiler = ModelProfiler(model_path, verbose=False, dtype=dtype, device=device)
        fwd_flops, bwd_flops, params = profiler.get_calflops(bs=bs, seq_len=seq_len, device=device)
        
        # Count actual total params from the model
        total_params = sum(p.numel() for p in profiler.model.parameters())
        total_params_b = total_params / 1e9
        
        result_queue.put({
            "status": "success",
            "fwd_flops": fwd_flops,
            "bwd_flops": bwd_flops,
            "params": params,
            "fwd_flops_gflops": parse_flops(fwd_flops),
            "params_b": parse_params(params),
            "total_params_b": round(total_params_b, 2),
            "layers": profiler.layer,
            "hidden_size": profiler.hidden_size,
            "dtype": str(profiler.dtype_),
        })
    except Exception as e:
        import traceback
        result_queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})


def run_time_test(model_path, bs, seq_len, device, dtype, skip_round, test_round, fwd_flag, block_flag, result_queue):
    """Test execution time (forward or forward+backward)"""
    try:
        import torch
        profiler = ModelProfiler(model_path, verbose=False, dtype=dtype, device=device)
        time_s, model_mem, act_mem, total_mem = profiler.profile(
            bs=bs, seq_len=seq_len, device=device,
            fwd_flag=fwd_flag, profile_flag='time',
            skip_round=skip_round, test_round=test_round,
            block_flag=block_flag
        )
        
        result_queue.put({
            "status": "success",
            "time_ms": round(time_s * 1000, 3),
            "fwd_flag": fwd_flag,
            "block_flag": block_flag,
        })
    except Exception as e:
        import traceback
        result_queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})


def run_memory_test(model_path, bs, seq_len, device, dtype, skip_round, test_round, fwd_flag, block_flag, result_queue):
    """Test memory"""
    try:
        import torch
        profiler = ModelProfiler(model_path, verbose=False, dtype=dtype, device=device)
        time_s, model_mem, act_mem, total_mem = profiler.profile(
            bs=bs, seq_len=seq_len, device=device,
            fwd_flag=fwd_flag, profile_flag='memory',
            skip_round=skip_round, test_round=test_round,
            block_flag=block_flag
        )
        
        result_queue.put({
            "status": "success",
            "model_mem_gb": round(model_mem, 4),
            "act_mem_gb": round(act_mem, 4),
            "total_mem_gb": round(total_mem, 4),
        })
    except Exception as e:
        import traceback
        result_queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})


def get_available_target_modules(trans_layer, preferred_modules):
    """Find available target modules from preferred list"""
    available = []
    layer_modules = set()
    
    # Get all module names from the transformer layer
    for name, _ in trans_layer.named_modules():
        if name:  # Skip the root module
            # Get the last part of the name
            parts = name.rsplit('.', 1)
            module_name = parts[-1] if len(parts) > 1 else name
            layer_modules.add(module_name)
    
    # Find which preferred modules exist
    for module in preferred_modules:
        if module in layer_modules:
            available.append(module)
    
    # If no preferred modules found, try common alternatives
    if not available:
        # Try to find any attention-related modules
        attention_keywords = ['q', 'k', 'v', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
        for module in layer_modules:
            for keyword in attention_keywords:
                if keyword in module.lower() and module not in available:
                    available.append(module)
                    break
    
    return available


def run_lora_flops_test(model_path, bs, seq_len, device, dtype, lora_config, lora_target_modules, result_queue):
    """Test LoRA fine-tuning FLOPs only"""
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        profiler = ModelProfiler(model_path, verbose=False, dtype=dtype, device=device)
        
        # Use model-specific target modules if provided, otherwise use default
        if lora_target_modules:
            target_modules = lora_target_modules
        else:
            preferred_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])
            target_modules = get_available_target_modules(profiler.trans, preferred_modules)
        
        if not target_modules:
            result_queue.put({
                "status": "error", 
                "error": f"No suitable target modules found. Available modules: {list(set(m for m, _ in profiler.trans.named_modules()))[:10]}"
            })
            return
        
        # Apply LoRA with available modules
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            target_modules=target_modules,
        )
        profiler.peftModel(peft_config)
        
        # Get FLOPs for LoRA model
        fwd_flops, bwd_flops, params = profiler.get_calflops(bs=bs, seq_len=seq_len, device=device)
        
        # Get trainable params info
        trainable_params = 0
        all_params = 0
        for _, param in profiler.trans.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        result_queue.put({
            "status": "success",
            "fwd_flops": fwd_flops,
            "bwd_flops": bwd_flops,
            "trainable_params": trainable_params * profiler.layer,
            "all_params": all_params * profiler.layer,
            "trainable_ratio": trainable_params / all_params if all_params > 0 else 0,
            "layers": profiler.layer,
        })
    except Exception as e:
        import traceback
        result_queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})


def run_lora_time_test(model_path, bs, seq_len, device, dtype, skip_round, test_round, lora_config, lora_target_modules, block_flag, result_queue):
    """Test LoRA fine-tuning time only"""
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        profiler = ModelProfiler(model_path, verbose=False, dtype=dtype, device=device)
        
        # Use model-specific target modules if provided, otherwise use default
        if lora_target_modules:
            target_modules = lora_target_modules
        else:
            preferred_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])
            target_modules = get_available_target_modules(profiler.trans, preferred_modules)
        
        if not target_modules:
            result_queue.put({
                "status": "error", 
                "error": f"No suitable target modules found. Available modules: {list(set(m for m, _ in profiler.trans.named_modules()))[:10]}"
            })
            return
        
        # Apply LoRA with available modules
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            target_modules=target_modules,
        )
        profiler.peftModel(peft_config)
        
        # Get actual dtype from model parameters
        actual_dtype = None
        for param in profiler.trans.parameters():
            if param is not None:
                actual_dtype = param.dtype
                break
        
        # Test backward time (training time = forward + backward)
        time_s, model_mem, act_mem, total_mem = profiler.profile(
            bs=bs, seq_len=seq_len, device=device,
            fwd_flag=False, profile_flag='time',
            skip_round=skip_round, test_round=test_round,
            skip_init=False, block_flag=block_flag
        )
        time_lora_ms = time_s * 1000
        
        result_queue.put({
            "status": "success",
            "time_fwd_bwd_ms": round(time_lora_ms, 3),
            "layers": profiler.layer,
            "dtype": str(actual_dtype) if actual_dtype else str(profiler.dtype_),
        })
    except Exception as e:
        import traceback
        result_queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})


def run_lora_memory_test(model_path, bs, seq_len, device, dtype, skip_round, test_round, lora_config, lora_target_modules, block_flag, result_queue):
    """Test LoRA fine-tuning memory only"""
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        profiler = ModelProfiler(model_path, verbose=False, dtype=dtype, device=device)
        
        # Use model-specific target modules if provided, otherwise use default
        if lora_target_modules:
            target_modules = lora_target_modules
        else:
            preferred_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])
            target_modules = get_available_target_modules(profiler.trans, preferred_modules)
        
        if not target_modules:
            result_queue.put({
                "status": "error", 
                "error": f"No suitable target modules found. Available modules: {list(set(m for m, _ in profiler.trans.named_modules()))[:10]}"
            })
            return
        
        # Apply LoRA with available modules
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            target_modules=target_modules,
        )
        profiler.peftModel(peft_config)
        
        # Test backward memory (training memory = forward + backward memory)
        time_s, model_mem, act_mem, total_mem = profiler.profile(
            bs=bs, seq_len=seq_len, device=device,
            fwd_flag=False, profile_flag='memory',
            skip_round=skip_round, test_round=test_round,
            skip_init=False, block_flag=block_flag
        )
        
        result_queue.put({
            "status": "success",
            "model_mem_gb": round(model_mem, 4),
            "act_mem_gb": round(act_mem, 4),
            "total_mem_gb": round(total_mem, 4),
        })
    except Exception as e:
        import traceback
        result_queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})


# ============================================================
# Wrapper Functions
# ============================================================

def test_flops(model_path, bs, seq_len, device, dtype, timeout=120):
    """Test FLOPs (subprocess)"""
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=run_flops_test,
        args=(model_path, bs, seq_len, device, dtype, result_queue)
    )
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "error", "error": "Timeout"}
    
    if not result_queue.empty():
        return result_queue.get()
    return {"status": "error", "error": "No result"}


def test_time(model_path, bs, seq_len, device, dtype, skip_round, test_round, fwd_flag=True, block_flag=False, timeout=180):
    """Test time (subprocess)"""
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=run_time_test,
        args=(model_path, bs, seq_len, device, dtype, skip_round, test_round, fwd_flag, block_flag, result_queue)
    )
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "error", "error": "Timeout"}
    
    if not result_queue.empty():
        return result_queue.get()
    return {"status": "error", "error": "No result"}


def test_memory(model_path, bs, seq_len, device, dtype, skip_round, test_round, fwd_flag=True, block_flag=False, timeout=180):
    """Test memory (subprocess)"""
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=run_memory_test,
        args=(model_path, bs, seq_len, device, dtype, skip_round, test_round, fwd_flag, block_flag, result_queue)
    )
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "error", "error": "Timeout"}
    
    if not result_queue.empty():
        return result_queue.get()
    return {"status": "error", "error": "No result"}


def test_lora_flops(model_path, bs, seq_len, device, dtype, lora_config, lora_target_modules=None, timeout=120):
    """Test LoRA fine-tuning FLOPs (subprocess)"""
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=run_lora_flops_test,
        args=(model_path, bs, seq_len, device, dtype, lora_config, lora_target_modules, result_queue)
    )
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "error", "error": "Timeout"}
    
    if not result_queue.empty():
        return result_queue.get()
    return {"status": "error", "error": "No result"}


def test_lora_time(model_path, bs, seq_len, device, dtype, skip_round, test_round, lora_config, lora_target_modules=None, block_flag=False, timeout=180):
    """Test LoRA fine-tuning time (subprocess)"""
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=run_lora_time_test,
        args=(model_path, bs, seq_len, device, dtype, skip_round, test_round, lora_config, lora_target_modules, block_flag, result_queue)
    )
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "error", "error": "Timeout"}
    
    if not result_queue.empty():
        return result_queue.get()
    return {"status": "error", "error": "No result"}


def test_lora_memory(model_path, bs, seq_len, device, dtype, skip_round, test_round, lora_config, lora_target_modules=None, block_flag=False, timeout=180):
    """Test LoRA fine-tuning memory (subprocess)"""
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=run_lora_memory_test,
        args=(model_path, bs, seq_len, device, dtype, skip_round, test_round, lora_config, lora_target_modules, block_flag, result_queue)
    )
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "error", "error": "Timeout"}
    
    if not result_queue.empty():
        return result_queue.get()
    return {"status": "error", "error": "No result"}


# ============================================================
# Main Function
# ============================================================

def run_benchmark():
    """Run full benchmark"""
    
    multiprocessing.set_start_method('spawn', force=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    bs = CONFIG["batch_size"]
    seq_len = CONFIG["seq_len"]
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": DEVICE,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(int(DEVICE.split(":")[1])) if torch.cuda.is_available() else "N/A",
            "config": CONFIG,
        },
        "models": {}
    }
    
    print("=" * 80)
    print("HF-LLM-Profiler Multi-Model Benchmark")
    print("Profile single Transformer Block, then scale to estimate full model")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(int(DEVICE.split(':')[1]))}")
    print(f"Batch Size: {bs}, Seq Len: {seq_len}")
    print()
    
    for model_info in MODELS:
        model_id = model_info["id"]
        model_name = model_info["name"]
        expected_params = model_info["params_b"]
        is_moe = model_info.get("is_moe", False)
        model_dtype = model_info.get("dtype", torch.bfloat16)
        
        print(f"\n{'='*80}")
        print(f"Testing: {model_name} ({model_id})")
        print(f"Expected Params: {expected_params}B")
        print(f"dtype: {model_dtype}")
        print(f"{'='*80}")
        
        model_results = {
            "name": model_name,
            "id": model_id,
            "expected_params_b": expected_params,
            "is_moe": is_moe,
            "batch_size": bs,
            "seq_len": seq_len,
            "dtype": str(model_dtype),
        }
        
        # 1. Download config.json
        print("\n[1/7] Downloading config.json...")
        try:
            model_path = download_config(model_id, MODEL_DIR)
            print(f"  ✓ Local path: {model_path}")
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            model_results["status"] = "download_error"
            model_results["error"] = str(e)
            results["models"][model_id] = model_results
            continue
        
        # 2. Test FLOPs
        print(f"\n[2/7] Testing FLOPs...")
        flops_result = test_flops(model_path, bs, seq_len, DEVICE, model_dtype, timeout=120)
        
        if flops_result["status"] == "success":
            model_results.update(flops_result)
            print(f"  ✓ FLOPs: {flops_result['fwd_flops']}")
            print(f"  ✓ Params (scaled): {flops_result['params']}")
            print(f"  ✓ Total Params (actual): {flops_result['total_params_b']} B")
            print(f"  ✓ Layers: {flops_result['layers']}, Hidden: {flops_result['hidden_size']}")
            # Parse backward FLOPs
            bwd_flops_g = parse_flops(flops_result['bwd_flops'])
            model_results["bwd_flops_gflops"] = bwd_flops_g
            print(f"  ✓ FLOPs: Forward {flops_result['fwd_flops']}, Backward {flops_result['bwd_flops']}")
        else:
            print(f"  ✗ FLOPs test failed: {flops_result.get('error', 'Unknown')}")
            model_results["flops_error"] = flops_result.get("error", "Unknown")
            if "traceback" in flops_result:
                print(f"     Traceback: {flops_result['traceback'][:200]}")
            results["models"][model_id] = model_results
            continue
        
        # 3. Test time (Forward) - block_flag=False to get estimated full model time directly
        print(f"\n[3/7] Testing time (Forward)...")
        time_fwd_result = test_time(
            model_path, bs, seq_len, DEVICE, model_dtype,
            CONFIG["skip_round"], CONFIG["test_round"], fwd_flag=True, block_flag=False, timeout=180
        )
        
        if time_fwd_result["status"] == "success":
            estimated_total_time_fwd_ms = time_fwd_result["time_ms"]
            layers = flops_result["layers"]
            block_time_fwd_ms = estimated_total_time_fwd_ms / layers
            model_results["block_time_fwd_ms"] = round(block_time_fwd_ms, 3)
            model_results["estimated_total_time_fwd_ms"] = round(estimated_total_time_fwd_ms, 2)
            model_results["estimated_total_time_fwd_s"] = round(estimated_total_time_fwd_ms / 1000, 3)
            print(f"  ✓ Forward Block Time: {block_time_fwd_ms:.3f} ms")
            print(f"  ✓ Estimated Full Model Forward Time ({layers} layers): {estimated_total_time_fwd_ms:.1f} ms ({estimated_total_time_fwd_ms/1000:.2f} s)")
        else:
            print(f"  ✗ Forward time test failed: {time_fwd_result.get('error', 'Unknown')}")
            model_results["time_fwd_error"] = time_fwd_result.get("error", "Unknown")
        
        # 4. Test time (Forward + Backward)
        print(f"\n[4/7] Testing time (Forward + Backward)...")
        time_bwd_result = test_time(
            model_path, bs, seq_len, DEVICE, model_dtype,
            CONFIG["skip_round"], CONFIG["test_round"], fwd_flag=False, block_flag=False, timeout=180
        )
        
        if time_bwd_result["status"] == "success":
            estimated_total_time_bwd_ms = time_bwd_result["time_ms"]
            layers = flops_result["layers"]
            block_time_bwd_ms = estimated_total_time_bwd_ms / layers
            model_results["block_time_fwd_bwd_ms"] = round(block_time_bwd_ms, 3)
            # Calculate backward only time
            block_time_bwd_only_ms = block_time_bwd_ms - model_results.get("block_time_fwd_ms", 0)
            model_results["block_time_bwd_only_ms"] = round(block_time_bwd_only_ms, 3)
            model_results["estimated_total_time_fwd_bwd_ms"] = round(estimated_total_time_bwd_ms, 2)
            model_results["estimated_total_time_fwd_bwd_s"] = round(estimated_total_time_bwd_ms / 1000, 3)
            print(f"  ✓ Forward+Backward Block Time: {block_time_bwd_ms:.3f} ms (Backward only: {block_time_bwd_only_ms:.3f} ms)")
            print(f"  ✓ Estimated Full Model Bwd Time ({layers} layers): {estimated_total_time_bwd_ms:.1f} ms ({estimated_total_time_bwd_ms/1000:.2f} s)")
        else:
            print(f"  ✗ Forward+Backward time test failed: {time_bwd_result.get('error', 'Unknown')}")
            model_results["time_bwd_error"] = time_bwd_result.get("error", "Unknown")
        
        # 5. Test memory (Forward)
        print(f"\n[5/7] Testing memory (Forward)...")
        mem_fwd_result = test_memory(
            model_path, bs, seq_len, DEVICE, model_dtype,
            CONFIG["skip_round"], CONFIG["test_round"], fwd_flag=True, block_flag=False, timeout=180
        )
        
        if mem_fwd_result["status"] == "success":
            estimated_total_mem_fwd = mem_fwd_result["total_mem_gb"]
            estimated_model_mem_fwd = mem_fwd_result["model_mem_gb"]
            estimated_act_mem_fwd = mem_fwd_result["act_mem_gb"]
            layers = flops_result["layers"]
            
            block_model_mem_fwd = estimated_model_mem_fwd / layers
            block_act_mem_fwd = estimated_act_mem_fwd / layers
            block_total_mem_fwd = estimated_total_mem_fwd / layers
            
            model_results["block_model_mem_fwd_gb"] = round(block_model_mem_fwd, 4)
            model_results["block_act_mem_fwd_gb"] = round(block_act_mem_fwd, 4)
            model_results["block_total_mem_fwd_gb"] = round(block_total_mem_fwd, 4)
            model_results["estimated_model_mem_fwd_gb"] = round(estimated_model_mem_fwd, 3)
            model_results["estimated_act_mem_fwd_gb"] = round(estimated_act_mem_fwd, 3)
            model_results["estimated_total_mem_fwd_gb"] = round(estimated_total_mem_fwd, 3)
            
            print(f"  ✓ Forward Memory (per block): Model {block_model_mem_fwd:.4f} GB, Activation {block_act_mem_fwd:.4f} GB, Total {block_total_mem_fwd:.4f} GB")
            print(f"  ✓ Estimated Full Model Forward Memory ({layers} layers): {estimated_total_mem_fwd:.2f} GB")
        else:
            print(f"  ✗ Forward memory test failed: {mem_fwd_result.get('error', 'Unknown')}")
            model_results["mem_fwd_error"] = mem_fwd_result.get("error", "Unknown")
        
        # 6. Test memory (Forward + Backward)
        print(f"\n[6/7] Testing memory (Forward + Backward)...")
        mem_bwd_result = test_memory(
            model_path, bs, seq_len, DEVICE, model_dtype,
            CONFIG["skip_round"], CONFIG["test_round"], fwd_flag=False, block_flag=False, timeout=180
        )
        
        if mem_bwd_result["status"] == "success":
            estimated_total_mem_bwd = mem_bwd_result["total_mem_gb"]
            estimated_model_mem_bwd = mem_bwd_result["model_mem_gb"]
            estimated_act_mem_bwd = mem_bwd_result["act_mem_gb"]
            layers = flops_result["layers"]
            
            block_model_mem_bwd = estimated_model_mem_bwd / layers
            block_act_mem_bwd = estimated_act_mem_bwd / layers
            block_total_mem_bwd = estimated_total_mem_bwd / layers
            
            model_results["block_model_mem_fwd_bwd_gb"] = round(block_model_mem_bwd, 4)
            model_results["block_act_mem_fwd_bwd_gb"] = round(block_act_mem_bwd, 4)
            model_results["block_total_mem_fwd_bwd_gb"] = round(block_total_mem_bwd, 4)
            model_results["estimated_model_mem_fwd_bwd_gb"] = round(estimated_model_mem_bwd, 3)
            model_results["estimated_act_mem_fwd_bwd_gb"] = round(estimated_act_mem_bwd, 3)
            model_results["estimated_total_mem_fwd_bwd_gb"] = round(estimated_total_mem_bwd, 3)
            
            # Backward only memory (activation gradients)
            act_mem_bwd_only = estimated_act_mem_bwd - model_results.get("estimated_act_mem_fwd_gb", 0)
            model_results["estimated_act_mem_bwd_only_gb"] = round(act_mem_bwd_only, 3)
            
            print(f"  ✓ Forward+Backward Memory (per block): Model {block_model_mem_bwd:.4f} GB, Activation {block_act_mem_bwd:.4f} GB, Total {block_total_mem_bwd:.4f} GB")
            print(f"  ✓ Estimated Full Model Bwd Memory ({layers} layers): {estimated_total_mem_bwd:.2f} GB (Backward activation: +{act_mem_bwd_only:.2f} GB)")
        else:
            print(f"  ✗ Forward+Backward memory test failed: {mem_bwd_result.get('error', 'Unknown')}")
            model_results["mem_bwd_error"] = mem_bwd_result.get("error", "Unknown")
        
        # 7. Test LoRA fine-tuning
        lora_target_modules = model_info.get("lora_target_modules", None)
        print(f"\n[7/7] Testing LoRA fine-tuning (r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}, targets={lora_target_modules or ['q_proj', 'v_proj']})...")
        
        # Test LoRA FLOPs first
        lora_flops_result = test_lora_flops(
            model_path, bs, seq_len, DEVICE, model_dtype,
            LORA_CONFIG, lora_target_modules, timeout=120
        )
        
        # Test LoRA time and memory separately to avoid memory measurement errors
        lora_time_result = test_lora_time(
            model_path, bs, seq_len, DEVICE, model_dtype,
            CONFIG["skip_round"], CONFIG["test_round"], LORA_CONFIG, lora_target_modules, block_flag=False, timeout=180
        )
        
        lora_memory_result = test_lora_memory(
            model_path, bs, seq_len, DEVICE, model_dtype,
            CONFIG["skip_round"], CONFIG["test_round"], LORA_CONFIG, lora_target_modules, block_flag=False, timeout=180
        )
        
        if lora_flops_result["status"] == "success" and lora_time_result["status"] == "success" and lora_memory_result["status"] == "success":
            model_results["lora_fwd_flops"] = lora_flops_result["fwd_flops"]
            model_results["lora_bwd_flops"] = lora_flops_result["bwd_flops"]
            model_results["lora_trainable_params"] = lora_flops_result["trainable_params"]
            model_results["lora_all_params"] = lora_flops_result["all_params"]
            model_results["lora_trainable_ratio"] = lora_flops_result["trainable_ratio"]
            model_results["lora_time_fwd_bwd_ms"] = lora_time_result["time_fwd_bwd_ms"]
            model_results["lora_model_mem_gb"] = lora_memory_result["model_mem_gb"]
            model_results["lora_act_mem_gb"] = lora_memory_result["act_mem_gb"]
            model_results["lora_total_mem_gb"] = lora_memory_result["total_mem_gb"]
            
            # With block_flag=False, results are already scaled to full model
            lora_estimated_time_s = lora_time_result["time_fwd_bwd_ms"] / 1000
            lora_full_model_mem = lora_memory_result["model_mem_gb"]
            lora_full_act_mem = lora_memory_result["act_mem_gb"]
            lora_full_total_mem = lora_memory_result["total_mem_gb"]
            
            layers = flops_result["layers"]
            # For per-block display, divide by layers
            lora_block_time_ms = lora_time_result["time_fwd_bwd_ms"] / layers
            lora_block_model_mem = lora_memory_result["model_mem_gb"] / layers
            lora_block_act_mem = lora_memory_result["act_mem_gb"] / layers
            lora_block_total_mem = lora_memory_result["total_mem_gb"] / layers
            
            print(f"  ✓ LoRA FLOPs: Forward {lora_flops_result['fwd_flops']}, Backward {lora_flops_result['bwd_flops']}")
            print(f"  ✓ LoRA Trainable Params: {lora_flops_result['trainable_params']/1e6:.2f}M / {lora_flops_result['all_params']/1e9:.2f}B ({lora_flops_result['trainable_ratio']*100:.2f}%)")
            print(f"  ✓ LoRA Block Time (Bwd): {lora_block_time_ms:.3f} ms")
            print(f"  ✓ Estimated Full Model LoRA Time: {lora_estimated_time_s:.2f} s")
            print(f"  ✓ LoRA Memory (per block): Model {lora_block_model_mem:.4f} GB, Activation {lora_block_act_mem:.4f} GB, Total {lora_block_total_mem:.4f} GB")
            print(f"  ✓ Estimated Full Model LoRA Memory ({layers} layers): {lora_full_total_mem:.2f} GB")
        else:
            if lora_flops_result["status"] != "success":
                print(f"  ✗ LoRA FLOPs test failed: {lora_flops_result.get('error', 'Unknown')}")
                model_results["lora_flops_error"] = lora_flops_result.get("error", "Unknown")
            if lora_time_result["status"] != "success":
                print(f"  ✗ LoRA time test failed: {lora_time_result.get('error', 'Unknown')}")
                model_results["lora_time_error"] = lora_time_result.get("error", "Unknown")
            if lora_memory_result["status"] != "success":
                print(f"  ✗ LoRA memory test failed: {lora_memory_result.get('error', 'Unknown')}")
                model_results["lora_memory_error"] = lora_memory_result.get("error", "Unknown")
        
        model_results["status"] = "success"
        model_results["lora_config"] = LORA_CONFIG
        results["models"][model_id] = model_results
        
        # Save intermediate results
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n\n{'='*140}")
    print(f"✅ Benchmark Complete! Results saved to: {OUTPUT_FILE}")
    print(f"{'='*140}")
    
    # Forward summary
    print(f"\n📊 Forward Pass Summary (Batch Size: {bs}, Seq Len: {seq_len}):")
    print(f"{'Model':<16} {'Active(B)':<10} {'Total(B)':<10} {'Layers':<6} {'Dtype':<10} {'Fwd FLOPs(T)':<12} {'Fwd Time(s)':<11} {'Fwd Mem(GB)':<11}")
    print("-" * 140)
    for mid, mdata in results["models"].items():
        if mdata.get("status") == "success":
            name = mdata.get("name", mid)[:14]
            active_params = mdata.get("params_b", 0)
            total_params = mdata.get("total_params_b", active_params)
            is_moe = mdata.get("is_moe", False)
            layers = mdata.get("layers", "N/A")
            dtype = mdata.get("dtype", "N/A").replace("torch.", "")
            flops_fwd = mdata.get("fwd_flops_gflops", 0) / 1000  # Convert GFLOPs to TFLOPs
            time_fwd_s = mdata.get("estimated_total_time_fwd_s", "N/A")
            mem_fwd = mdata.get("estimated_total_mem_fwd_gb", "N/A")
            time_str = f"{time_fwd_s:.2f}" if isinstance(time_fwd_s, (int, float)) else str(time_fwd_s)
            mem_str = f"{mem_fwd:.1f}" if isinstance(mem_fwd, (int, float)) else str(mem_fwd)
            name_display = f"{name}*" if is_moe else name
            print(f"{name_display:<16} {active_params:<10.2f} {total_params:<10.2f} {layers:<6} {dtype:<10} {flops_fwd:<12.2f} {time_str:<11} {mem_str}")
    
    # Forward+Backward summary
    print(f"\n📊 Forward + Backward Pass Summary (Batch Size: {bs}, Seq Len: {seq_len}):")
    print(f"{'Model':<16} {'Active(B)':<10} {'Total(B)':<10} {'Layers':<6} {'Dtype':<10} {'Bwd FLOPs(T)':<12} {'Bwd Time(s)':<14} {'Bwd Mem(GB)':<14}")
    print("-" * 140)
    for mid, mdata in results["models"].items():
        if mdata.get("status") == "success":
            name = mdata.get("name", mid)[:14]
            active_params = mdata.get("params_b", 0)
            total_params = mdata.get("total_params_b", active_params)
            is_moe = mdata.get("is_moe", False)
            layers = mdata.get("layers", "N/A")
            dtype = mdata.get("dtype", "N/A").replace("torch.", "")
            flops_bwd = mdata.get("bwd_flops_gflops", 0) / 1000  # Convert GFLOPs to TFLOPs
            time_bwd_s = mdata.get("estimated_total_time_fwd_bwd_s", "N/A")
            mem_bwd = mdata.get("estimated_total_mem_fwd_bwd_gb", "N/A")
            time_str = f"{time_bwd_s:.2f}" if isinstance(time_bwd_s, (int, float)) else str(time_bwd_s)
            mem_str = f"{mem_bwd:.1f}" if isinstance(mem_bwd, (int, float)) else str(mem_bwd)
            name_display = f"{name}*" if is_moe else name
            print(f"{name_display:<16} {active_params:<10.2f} {total_params:<10.2f} {layers:<6} {dtype:<10} {flops_bwd:<12.2f} {time_str:<14} {mem_str}")
    
    # LoRA Fine-tuning summary
    print(f"\n📊 LoRA Fine-tuning Summary (r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}, dropout={LORA_CONFIG['lora_dropout']}, Batch Size: {bs}, Seq Len: {seq_len}):")
    print(f"{'Model':<16} {'Trainable(M)':<12} {'All Params(B)':<14} {'Train(%)':<8} {'Bwd FLOPs(T)':<12} {'Bwd Time(s)':<14} {'Memory(GB)':<12}")
    print("-" * 160)
    for mid, mdata in results["models"].items():
        if mdata.get("status") == "success" and mdata.get("lora_trainable_params"):
            name = mdata.get("name", mid)[:14]
            trainable = mdata.get("lora_trainable_params", 0) / 1e6  # Convert to millions
            all_params = mdata.get("total_params_b", mdata.get("lora_all_params", 0) / 1e9)  # Use model's total params
            train_ratio = mdata.get("lora_trainable_ratio", 0) * 100
            # Parse FLOPs
            bwd_flops_str = mdata.get("lora_bwd_flops", "N/A")
            bwd_flops_t = parse_flops(bwd_flops_str) / 1000 if bwd_flops_str != "N/A" else "N/A"  # Convert to TFLOPs
            time_s = mdata.get("lora_time_fwd_bwd_ms", 0) / 1000  # Full model time
            mem_gb = mdata.get("lora_total_mem_gb", 0)
            is_moe = mdata.get("is_moe", False)
            name_display = f"{name}*" if is_moe else name
            bwd_flops_str_out = f"{bwd_flops_t:.2f}" if isinstance(bwd_flops_t, (int, float)) else str(bwd_flops_t)
            print(f"{name_display:<16} {trainable:<12.2f} {all_params:<14.2f} {train_ratio:<8.2f} {bwd_flops_str_out:<12} {time_s:<14.2f} {mem_gb:<12.2f}")
    
    print(f"\n* MoE models: Active params = activated params per forward, Total params = all expert weights")
    print(f"  Fwd = Forward only, Bwd = Forward + Backward (training mode)")
    print(f"  Time and memory results are estimated for the full model based on single block measurements.")
    print(f"  LoRA trainable is a litter bigger than truth and flops/times/memory is a little smaller than truth.")
    return results


if __name__ == "__main__":
    run_benchmark()