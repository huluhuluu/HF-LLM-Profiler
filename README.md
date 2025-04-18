# HF-LLM-Profiler
Wonder know how much GPU memory, time and compute flops running a Huggingface LLM requires? Profile it!

Considering that running a LLM in practice requires massive GPU memory and computational resources, which we may not always have. Since the bulk of computation in LLM occurs in uniformly stacked transformer blocks, we estimate the total GPU memory, runtime, and FLOPs for the full model by **scaling up the results from one block**. 

# 1. Environment
```shell
pip install -r requirements.txt 
pip install calculate_flops
```

# 2. DownLoad config
## 2.1 from hf-mirror/hf
Using [hfd](https://hf-mirror.com/) tool to download model's config.
```shell
./hfd.sh model_id --include '^config.json$'
# eg: ./hfd.sh google-bert/bert-base-uncased --include '^config.json$'
# eg: ./hfd.sh openai-community/gpt2 --include '^config.json$'
```
## 2.2 directly from hf-hub
Set ```model_id_or_path``` to model_id in hf-hub and make sure network connectivity.

# 3. Run Profile
## 3.1 set parameter
| Parameter           | Meaning                                         |
|:-------------------:|:-----------------------------------------------:|
| model\_id\_or\_path | model id in hf\-hub or downloaded in local path |
| bs                  | batch size to profile                           |
| seq\_len            | sequence length to profile                      |
| device              | device to run the model on                      |
| fwd_flag            | True for forward pass, False for backward pass  |

## 3.1 profile memory
```python
model_id_or_path = '/data/HUGGINGFACE/Falcon3-10B-Base' # Qwen2.5-3B Falcon3-10B-Base
profiler = ModelProfiler(model_id_or_path)
# profile memory
profiler.profile(bs = 8, seq_len = 512, device = 'cuda:0', skip_round=100, test_round=500, fwd_flag=False, profile_flag='memory')
```
## 3.2 profile time
```python
model_id_or_path = '/data/HUGGINGFACE/Falcon3-10B-Base' # Qwen2.5-3B Falcon3-10B-Base
profiler = ModelProfiler(model_id_or_path)
# profile time
profiler.profile(bs = 8, seq_len = 512, device = 'cuda:0', skip_round=100, test_round=500, fwd_flag=False, profile_flag='time')
```
## 3.3 profile flops
```python
model_id_or_path = '/data/HUGGINGFACE/Falcon3-10B-Base' # Qwen2.5-3B Falcon3-10B-Base
profiler = ModelProfiler(model_id_or_path)
# profile flops
profiler.get_calflops(bs = 8, seq_len = 512, device = 'cuda:0')
```
# 4.