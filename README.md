# 🔍 HF-LLM-Profiler

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**English** | **[中文](README_CN.md)**

> 🚀 **Want to know how much GPU memory, time, and FLOPs are needed to run Hugging Face LLMs? Analyze with one click!**
> 
> 📊 **Want to understand model architecture? Print it directly!**

## ✨ Core Features

| Feature | Description |
|------|------|
| 🎯 **Smart Estimation** | Estimate full model resource requirements by scaling from a single Transformer Block |
| 📈 **Multi-dimensional Analysis** | Support GPU memory, execution time, FLOPs three analysis dimensions |
| 🔧 **Flexible Configuration** | Support custom batch size, sequence length, device type |
| 🤗 **Wide Compatibility** | Support mainstream LLMs from Hugging Face model hub |
| 🔌 **PEFT Support** | Built-in LoRA fine-tuning configuration support |
| ⚡ **Multi-device** | Support CUDA and NPU (Huawei Atlas) devices |

## ⚠️ Important Notes

| Note | Description |
|----------|------|
| 📊 **Estimation Error** | Based on single block scaling, may have some error, especially for MoE models |
| 🏗️ **Architecture Limitations** | Some models may not work due to special architectures, some models may need manual model_config addition |
| 🔥 **Warm-up Required** | Need warm-up and multiple test rounds for stable results, each profile time and memory should be started through a process, and cannot be tested continuously in one function, to avoid memory error caused by model already on target device |

## 📦 Installation

```bash
# Option 1: Direct install
pip install -r requirements.txt

# Option 2: Conda virtual environment (recommended)
conda create -n llm-profiler python=3.10 -y
conda activate llm-profiler
pip install -r requirements.txt
```

**Main Dependencies**: `torch` | `transformers` | `peft` | `calflops`

## 🚀 Quick Start

### 1. Get Model Config

```python
# Option 1: Use HF Hub model ID directly (requires network)
model_id = "Qwen/Qwen2.5-0.5B"

# Option 2: Use hfd tool to download config (offline)
# ./hfd.sh <model_id> --include '^config.json$'
```

### 2. Print Model Structure

```python
from Profiler import Profiler

profiler = Profiler('bert-base-uncased')
print(profiler.model)
print(f"Layers: {profiler.layer}, Hidden Size: {profiler.hidden_size}")
```

### 3. Analyze FLOPs

```python
from Profiler import ModelProfiler

profiler = ModelProfiler('Qwen2.5-0.5B', verbose=True)
fwd_flops, bwd_flops, params = profiler.get_calflops(bs=8, seq_len=512, device='cuda:0')

# Output example:
# GPT-2 | batch size 8 | seq_len 512 | layers 12 | torch.float32
# forward flops: 22.56 GFLOPS | backward flops: 67.68 GFLOPS | params: 124.44 M
```

### 4. Analyze Memory Usage

```python
profiler = ModelProfiler('Qwen2.5-0.5B', verbose=True)

# Forward pass memory
profiler.profile(bs=8, seq_len=512, device='cuda:0', 
                 fwd_flag=True, profile_flag='memory',
                 skip_round=100, test_round=500)

# Forward+Backward pass memory
profiler.profile(bs=8, seq_len=512, device='cuda:0',
                 fwd_flag=False, profile_flag='memory',
                 skip_round=100, test_round=500)
```

### 5. Analyze Execution Time

```python
profiler = ModelProfiler('Qwen2.5-0.5B', verbose=True)

# Single Block execution time
fwd_time, model_mem, act_mem, total_mem = profiler.profile(
    bs=8, seq_len=512, device='cuda:0',
    fwd_flag=True, profile_flag='time',
    skip_round=100, test_round=500
)
print(f"Single block forward time: {fwd_time:.5f}s")
```

## 📊 Parameter Description

| Parameter | Type | Description | Default |
|------|------|------|--------|
| `model_id_or_path` | str | HF Hub model ID or local path | Required |
| `bs` | int | Batch Size | 8 |
| `seq_len` | int | Sequence length | 512 |
| `device` | str | Running device | `cuda:0` |
| `fwd_flag` | bool | `True`=forward only, `False`=forward+backward | `True` |
| `profile_flag` | str | `time`=time, `memory`=memory | `time` |
| `skip_round` | int | Warm-up rounds (not counted in statistics) | 100 |
| `test_round` | int | Test rounds | 500 |
| `dtype` | torch.dtype | Data type | Model default |
| `skip_init` | bool | Whether to skip model initialization (model already on target device) | `False` |
| `block_flag` | bool | Whether the return value is a single Block or the full model | `True` |
| `config` | PeftConfig | PEFT compatible fine-tuning parameters | `None |
| `verbose` | bool | Whether to print detailed information | `False` |

## 🔧 Advanced Usage

### LoRA Fine-tuning Support

```python
from Profiler import ModelProfiler
from peft import LoraConfig

# Create LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias='none',
    target_modules=['q_proj', 'v_proj']
)

profiler = ModelProfiler('meta-llama/Llama-2-7b-hf', verbose=True)
profiler.peftModel(lora_config)

# Analyze fine-tuned model
profiler.get_calflops(bs=4, seq_len=256, device='cuda:0')
```

### Specify Data Type

```python
import torch

# Use FP16
profiler = ModelProfiler('Qwen2.5-0.5B', dtype=torch.float16, verbose=True)

# Use BF16 (requires Ampere+ GPU)
profiler = ModelProfiler('Qwen2.5-0.5B', dtype=torch.bfloat16, verbose=True)
```

### NPU Device Support (Huawei Atlas)

```python
profiler = ModelProfiler('Qwen2.5-0.5B', device='npu:0', verbose=True)
profiler.profile(bs=8, seq_len=512, device='npu:0', ...)
```

## Run Test
### Configure Parameters
Modify test parameters in test function in test.py:
```python
# Corresponding model parameters, running gpu, and whether to test LoRA fine-tuning
path, device, test_lora = '/workspace/code/HF-LLM-Profiler/models/Qwen2.5-0.5B', "cuda:5", True
```
### Run Test
```bash
# Activate virtual environment and run single model test
> python test.py
------------------Qwen2.5-0.5B------------------
Qwen2Model(
  (embed_tokens): Embedding(151936, 896)
  (layers): ModuleList(
    (0-23): 24 x Qwen2DecoderLayer(
      (self_attn): Qwen2Attention(
        (q_proj): Linear(in_features=896, out_features=896, bias=True)
        (k_proj): Linear(in_features=896, out_features=128, bias=True)
        (v_proj): Linear(in_features=896, out_features=128, bias=True)
        (o_proj): Linear(in_features=896, out_features=896, bias=False)
      )
      (mlp): Qwen2MLP(
        (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
        (up_proj): Linear(in_features=896, out_features=4864, bias=False)
        (down_proj): Linear(in_features=4864, out_features=896, bias=False)
        (act_fn): SiLUActivation()
      )
      (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
      (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
    )
  )
  (norm): Qwen2RMSNorm((896,), eps=1e-06)
  (rotary_emb): Qwen2RotaryEmbedding()
)
------------------Qwen2.5-0.5B Profile flops------------------
Qwen2.5-0.5B | batch size 8 | seq_len 512 | layers 24 | torch.bfloat16
forward flops: 2936.16 GFLOPS | backward flops: 8808.72 GFLOPS | params: 540.72 K
------------------Qwen2.5-0.5B Profile memory------------------
Qwen2.5-0.5B | batch size 8 | seq_len 512 | layers 24 | torch.bfloat16 | forward
model memory: 0.0279/0.6687 GB
activation memory: 0.1407/3.3779 GB
total memory: 0.1686/4.0466 GB
Qwen2.5-0.5B | batch size 8 | seq_len 512 | layers 24 | torch.bfloat16 | backward
model memory: 0.0279/0.6687 GB
activation memory: 0.3494/8.3848 GB
total memory: 0.3772/9.0534 GB
------------------Qwen2.5-0.5B Profile time------------------
Qwen2.5-0.5B | batch size 8 | seq_len 512 | layers 24 | torch.bfloat16 | forward
model runing time: 0.06006 s
Qwen2.5-0.5B | batch size 8 | seq_len 512 | layers 24 | torch.bfloat16 | backward
model runing time: 0.16165 s
```

## 🧪 Run Benchmark
### Configuration
Configure benchmark parameters in Configuration at the beginning of benchmark.py:
```python
# benchmark.py
# ============================================================
# Configuration
# ============================================================

# Test device configuration, support CUDA and Atlas NPU
# Device configuration
DEVICE = "cuda:5"

# Test data batch size and sequence length, and warm-up rounds and test rounds
# Test configuration
CONFIG = {
    "batch_size": 8,
    "seq_len": 512,
    "skip_round": 5,
    "test_round": 10,
}

# LoRA fine-tuning test parameters
# LoRA configuration for fine-tuning tests
LORA_CONFIG = {
    "r": 8,                    # LoRA rank
    "lora_alpha": 16,          # LoRA alpha scaling
    "lora_dropout": 0.05,      # Dropout probability
    "bias": "none",            # Bias type
    "target_modules": ["q_proj", "v_proj"],  # Target attention modules
}

# HFD downloader path
HFD_SCRIPT = "/data/HUGGINGFACE/hfd.sh" # huggingface-cli tool for downloading model config

# Model save directory
MODEL_DIR = "/workspace/code/HF-LLM-Profiler/models"  # Model storage directory

# Output file
OUTPUT_FILE = "bench/measured_data.json"  # benchmark result output file

# Hugging Face credentials (for gated models)
HF_USERNAME = "huluhuluu"   # Hugging Face username
HF_TOKEN = "hf_ababababababababababababab" # Hugging Face access token

# Models below can be freely added
# ============================================================
# Test Model List
# Ranging from ~0.5B to 100B+ parameters
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

```

### Run
```bash
# Run multi-model benchmark
python benchmark.py

# Generate visualization charts
python generate_charts.py
```

### Benchmark Results
```bash
============================================================================================================================================
✅ Benchmark Complete! Results saved to: bench/measured_data.json
============================================================================================================================================

📊 Forward Pass Summary (Batch Size: 8, Seq Len: 512):
Model            Active(B)  Total(B)   Layers Dtype      Fwd FLOPs(T) Fwd Time(s) Fwd Mem(GB)
--------------------------------------------------------------------------------------------------------------------------------------------
Qwen2.5-0.5B     0.36       0.49       24     bfloat16   2.93         0.06        4.0
Qwen2.5-1.5B     1.31       1.54       28     bfloat16   10.73        0.13        9.4
Phi-2            2.52       2.65       32     bfloat16   20.62        0.27        16.8
Qwen2.5-3B       2.77       3.09       36     bfloat16   22.73        0.24        16.4
Qwen2.5-7B       6.53       7.07       28     bfloat16   53.48        0.48        26.9
Llama-3.2-3B     2.82       3.21       28     bfloat16   23.09        0.27        12.7
Llama-3-8B       6.98       7.50       32     bfloat16   57.28        0.54        26.8
Mistral-7B       6.98       7.11       32     bfloat16   57.28        0.54        26.8
Qwen2.5-14B      13.21      13.99      48     bfloat16   108.00       0.98        46.0
Qwen2.5-32B      31.21      31.99      64     bfloat16   255.36       1.97        106.7
Llama-3.1-70B    68.45      69.50      80     bfloat16   560.80       4.12        195.6
Qwen2.5-72B      70.21      71.46      80     bfloat16   575.20       4.24        200.9
DeepSeek-V3*     35.59      670.10     61     bfloat16   291.58       2.43        150.1

📊 Forward + Backward Pass Summary (Batch Size: 8, Seq Len: 512):
Model            Active(B)  Total(B)   Layers Dtype      Bwd FLOPs(T) Bwd Time(s)    Bwd Mem(GB)
--------------------------------------------------------------------------------------------------------------------------------------------
Qwen2.5-0.5B     0.36       0.49       24     bfloat16   8.80         0.13           9.1
Qwen2.5-1.5B     1.31       1.54       28     bfloat16   32.20        0.35           20.0
Phi-2            2.52       2.65       32     bfloat16   61.76        0.70           30.2
Qwen2.5-3B       2.77       3.09       36     bfloat16   68.04        0.69           34.0
Qwen2.5-7B       6.53       7.07       28     bfloat16   160.44       1.28           52.0
Llama-3.2-3B     2.82       3.21       28     bfloat16   69.16        0.67           27.4
Llama-3-8B       6.98       7.50       32     bfloat16   171.52       1.44           53.0
Mistral-7B       6.98       7.11       32     bfloat16   171.52       1.41           53.0
Qwen2.5-14B      13.21      13.99      48     bfloat16   324.48       2.69           90.5
Qwen2.5-32B      31.21      31.99      64     bfloat16   766.72       5.60           194.6
Llama-3.1-70B    68.45      69.50      80     bfloat16   1682.40      11.72          348.9
Qwen2.5-72B      70.21      71.46      80     bfloat16   1725.60      11.98          355.5
DeepSeek-V3*     35.59      670.10     61     bfloat16   874.74       7.26           268.9

📊 LoRA Fine-tuning Summary (r=8, alpha=16, dropout=0.05, Batch Size: 8, Seq Len: 512):
Model            Trainable(M) All Params(B)  Train(%) Bwd FLOPs(T) Bwd Time(s)    Memory(GB)
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Qwen2.5-0.5B     0.54         0.49           0.15     8.81         0.12           9.05
Qwen2.5-1.5B     1.09         1.54           0.08     32.20        0.31           19.66
Phi-2            2.62         2.65           0.10     61.76        0.61           30.57
Qwen2.5-3B       1.84         3.09           0.07     68.40        0.58           33.05
Qwen2.5-7B       2.52         7.07           0.04     160.44       1.02           49.23
Llama-3.2-3B     2.29         3.21           0.08     69.44        0.60           26.72
Llama-3-8B       3.41         7.50           0.05     171.52       1.19           50.55
Mistral-7B       3.41         7.11           0.05     171.52       1.17           50.55
Qwen2.5-14B      6.29         13.99          0.05     324.96       2.24           86.00
Qwen2.5-32B      8.39         31.99          0.03     767.36       4.35           180.29
Llama-3.1-70B    16.38        69.50          0.02     1682.40      9.21           313.93
Qwen2.5-72B      16.38        71.46          0.02     1725.60      9.22           321.04
DeepSeek-V3*     16.99        670.10         0.05     875.35       6.16           212.17

* MoE models: Active params = activated params per forward, Total params = all expert weights
  Fwd = Forward only, Bwd = Forward + Backward (training mode)
  Time and memory results are estimated for the full model based on single block measurements.
  LoRA trainable is a litter bigger than truth and flops/times/memory is a little smaller than truth.
```
### Visualization Results
Generated charts are saved in bench directory:
- `bench/fwd_bwd_time_comparison.png`: Forward and backward pass estimated time comparison
![fwd_bwd_time](/bench/fwd_bwd_time_comparison.png)
- `bench/fwd_bwd_memory_comparison.png`: Backward pass memory and compute graph
![fwd_bwd_memory](/bench/training_flops_vs_memory.png)

## 🏗️ Project Structure

```
HF-LLM-Profiler/
├── Profiler.py          # Core profiler implementation 
├── model_config.py      # Model configuration definition
├── flops_counter.py     # FLOPs calculation module
├── test.py              # Test script
├── benchmark.py         # Multi-model benchmark script
├── plot_from_json.py    # Chart generation script
├── requirements.txt     # Dependency list
├── README.md            # Project documentation
└── bench/   # Benchmark output directory
```

## ⚙️ Working Principle

This tool adopts **single block scaling estimation** strategy:

1. **Load Config** - Load model config from HF Hub or local (without loading weights)
2. **Build Empty Model** - Create model on meta device using `accelerate.init_empty_weights`
3. **Extract Single Block** - Get a single Transformer Block
4. **Actual Test** - Run this Block on target device
5. **Scale Estimation** - Multiply results by number of layers to get full model estimation

Advantages of this method:
- ✅ No need to download complete model weights
- ✅ Fast test speed
- ✅ Low memory usage
- ✅ Applicable to models of any size

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [calflops](https://github.com/MrYxJ/calculate-flops.pytorch)
- [PEFT](https://github.com/huggingface/peft)
- [IFLOW](https://github.com/iflow-ai)