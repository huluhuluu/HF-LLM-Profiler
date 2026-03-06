# 🔍 HF-LLM-Profiler

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**[English](README.md)** | **中文**

> 🚀 **想知道运行 Hugging Face LLM 需要多少 GPU 内存、时间和计算 FLOPs？一键分析！**
> 
> 📊 **想了解模型结构？直接打印！**

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🎯 **智能估算** | 通过单个 Transformer Block 缩放估算全模型资源需求 |
| 📈 **多维度分析** | 支持 GPU 内存、执行时间、FLOPs 三种分析维度 |
| 🔧 **灵活配置** | 支持自定义 batch size、序列长度、设备类型 |
| 🤗 **广泛兼容** | 支持 Hugging Face 模型库中的主流 LLM |
| 🔌 **PEFT 支持** | 内置 LoRA 微调配置支持 |
| ⚡ **多设备** | 支持 CUDA 和 NPU（华为 Atlas）设备 |

## ⚠️ 注意事项

| 注意事项 | 说明 |
|----------|------|
| 📊 **估算误差** | 基于单块缩放，可能存在一定误差，尤其是 MoE 模型 |
| 🏗️ **架构限制** | 部分模型可能因架构特殊而不适用， 部分模型可能需要手动添加model_config |
| 🔥 **预热需求** | 需要预热和多轮测试以获得稳定结果，每次profile时间和显存时通过进程启动，并且不能在一个函数中连续测试，避免模型已经在目标设备导致显存误差 |

## 📦 安装

```bash
# 方式一：直接安装
pip install -r requirements.txt

# 方式二：Conda 虚拟环境（推荐）
conda create -n llm-profiler python=3.10 -y
conda activate llm-profiler
pip install -r requirements.txt
```

**主要依赖**：`torch` | `transformers` | `peft` | `calflops`

## 🚀 快速开始

### 1. 获取模型配置

```python
# 方式一：直接使用 HF Hub 模型 ID（需要网络）
model_id = "Qwen/Qwen2.5-0.5B"

# 方式二：使用 hfd 工具下载配置（离线可用）
# ./hfd.sh <model_id> --include '^config.json$'
```

### 2. 打印模型结构

```python
from Profiler import Profiler

profiler = Profiler('bert-base-uncased')
print(profiler.model)
print(f"Layers: {profiler.layer}, Hidden Size: {profiler.hidden_size}")
```

### 3. 分析 FLOPs

```python
from Profiler import ModelProfiler

profiler = ModelProfiler('Qwen2.5-0.5B', verbose=True)
fwd_flops, bwd_flops, params = profiler.get_calflops(bs=8, seq_len=512, device='cuda:0')

# 输出示例：
# GPT-2 | batch size 8 | seq_len 512 | layers 12 | torch.float32
# forward flops: 22.56 GFLOPS | backward flops: 67.68 GFLOPS | params: 124.44 M
```

### 4. 分析内存使用

```python
profiler = ModelProfiler('Qwen2.5-0.5B', verbose=True)

# 前向传播内存
profiler.profile(bs=8, seq_len=512, device='cuda:0', 
                 fwd_flag=True, profile_flag='memory',
                 skip_round=100, test_round=500)

# 前向+反向传播内存
profiler.profile(bs=8, seq_len=512, device='cuda:0',
                 fwd_flag=False, profile_flag='memory',
                 skip_round=100, test_round=500)
```

### 5. 分析执行时间

```python
profiler = ModelProfiler('Qwen2.5-0.5B', verbose=True)

# 单 Block 执行时间
fwd_time, model_mem, act_mem, total_mem = profiler.profile(
    bs=8, seq_len=512, device='cuda:0',
    fwd_flag=True, profile_flag='time',
    skip_round=100, test_round=500
)
print(f"Single block forward time: {fwd_time:.5f}s")
```

## 📊 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `model_id_or_path` | str | HF Hub 模型 ID 或本地路径 | 必填 |
| `bs` | int | Batch Size | 8 |
| `seq_len` | int | 序列长度 | 512 |
| `device` | str | 运行设备 | `cuda:0` |
| `fwd_flag` | bool | `True`=仅前向，`False`=前向+反向 | `True` |
| `profile_flag` | str | `time`=时间，`memory`=内存 | `time` |
| `skip_round` | int | 预热轮次（不计入统计） | 100 |
| `test_round` | int | 测试轮次 | 500 |
| `dtype` | torch.dtype | 数据类型 | 模型默认 |


## 🔧 高级用法

### LoRA 微调支持

```python
from Profiler import ModelProfiler
from peft import LoraConfig

# 创建 LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias='none',
    target_modules=['q_proj', 'v_proj']
)

profiler = ModelProfiler('meta-llama/Llama-2-7b-hf', verbose=True)
profiler.peftModel(lora_config)

# 分析微调后的模型
profiler.get_calflops(bs=4, seq_len=256, device='cuda:0')
```

### 指定数据类型

```python
import torch

# 使用 FP16
profiler = ModelProfiler('Qwen2.5-0.5B', dtype=torch.float16, verbose=True)

# 使用 BF16 (需要 Ampere+ GPU)
profiler = ModelProfiler('Qwen2.5-0.5B', dtype=torch.bfloat16, verbose=True)
```

### NPU 设备支持（华为 Atlas）

```python
profiler = ModelProfiler('Qwen2.5-0.5B', device='npu:0', verbose=True)
profiler.profile(bs=8, seq_len=512, device='npu:0', ...)
```

## 运行测试
### 配置参数
在test.py中的test函数中修改测试参数：
```python
# 对应模型参数 运行gpu 以及是否测试LoRA微调
path, device, test_lora = '/workspace/code/HF-LLM-Profiler/models/Qwen2.5-0.5B', "cuda:5", True
```
### 运行测试
```bash
# 启动虚拟环境 并且 运行单模型测试
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

## 🧪 运行 Benchmark
### 配置
配置benchmark参数，在benchmark.py代码开头的Configuration中进行修改：
```python
# benchmark.py
# ============================================================
# Configuration
# ============================================================

# 测试设备配置，支持 CUDA 和 Atlas NPU
# Device configuration
DEVICE = "cuda:5"

# 测试的数据批大小和序列长度 以及 预热轮次和测试轮次
# Test configuration
CONFIG = {
    "batch_size": 8,
    "seq_len": 512,
    "skip_round": 5,
    "test_round": 10,
}

# LoRA微调的测试参数
# LoRA configuration for fine-tuning tests
LORA_CONFIG = {
    "r": 8,                    # LoRA rank
    "lora_alpha": 16,          # LoRA alpha scaling
    "lora_dropout": 0.05,      # Dropout probability
    "bias": "none",            # Bias type
    "target_modules": ["q_proj", "v_proj"],  # Target attention modules
}

# HFD downloader path
HFD_SCRIPT = "/data/HUGGINGFACE/hfd.sh" # huggingface-cli 工具，用于下载模型配置

# Model save directory
MODEL_DIR = "/workspace/code/HF-LLM-Profiler/models"  # 模型存放目录

# Output file
OUTPUT_FILE = "bench/measured_data.json"  # benchmark结果输出文件

# Hugging Face credentials (for gated models)
HF_USERNAME = "huluhuluu"   # Hugging Face 用户名
HF_TOKEN = "hf_ababababababababababababab" # Hugging Face 访问令牌

# 下面模型可以自由添加
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

### 运行
```bash
# 运行多模型 benchmark
python benchmark.py

# 生成可视化图表
python generate_charts.py
```

### bench结果
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
### 可视化结果
生成的图表保存在bench目录下：
- `bench/fwd_bwd_time_comparison.png`: 前向传播和反向传播的预估时间比较
![fwd_bwd_time](/bench/fwd_bwd_time_comparison.png)
- `bench/fwd_bwd_memory_comparison.png`: 反向传播占用的显存与计算量图
![fwd_bwd_memory](/bench/fwd_bwd_memory_comparison.png)

## 🏗️ 项目结构

```
HF-LLM-Profiler/
├── Profiler.py          # 核心分析器实现 
├── model_config.py      # 模型配置定义
├── flops_counter.py     # FLOPs 计算模块
├── test.py              # 测试脚本
├── benchmark.py         # 多模型 benchmark 脚本
├── plot_from_json.py    # 图表生成脚本
├── requirements.txt     # 依赖列表
├── README.md            # 项目文档
└── bench/   # Benchmark 输出目录
```

## ⚙️ 工作原理

本工具采用**单块缩放估算**策略：

1. **加载配置** - 从 HF Hub 或本地加载模型配置（不加载权重）
2. **构建空模型** - 使用 `accelerate.init_empty_weights` 创建 meta 设备上的模型
3. **提取单块** - 获取单个 Transformer Block
4. **实际测试** - 在目标设备上运行该 Block
5. **缩放估算** - 将结果乘以层数得到全模型估算值

这种方法的优势：
- ✅ 无需下载完整模型权重
- ✅ 测试速度快
- ✅ 内存占用低
- ✅ 适用于任意大小的模型

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [calflops](https://github.com/MrYxJ/calculate-flops.pytorch)
- [PEFT](https://github.com/huggingface/peft)