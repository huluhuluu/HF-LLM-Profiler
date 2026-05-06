import torch
import multiprocessing
import sys
from Profiler import Profiler, ModelProfiler, EmbeddingProfiler, FFNProfiler
try:
    from peft import LoraConfig
except ModuleNotFoundError:
    LoraConfig = None

def clear_device_cache(device):
    if 'cuda' in device and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif 'npu' in device and hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()

def print_model(path = '/workspace/code/HF-LLM-Profiler/models/Meta-Llama-3.1-70B'):
    '''
        Print the model's structure w/o donwload hole model.
    '''
    profiler = Profiler(path, verbose=True)
    print(f'------------------{profiler.model_id}------------------')
    print(profiler.model)

def test_flops(path = '/workspace/code/HF-LLM-Profiler/models/Meta-Llama-3.1-70B', bs = 8, seq = 512, device = 'cuda:0', test_lora = False):
    '''
        Test the flops of the model.
    '''
    profiler = ModelProfiler(path, verbose=True)
    if test_lora:
        if LoraConfig is None:
            raise ModuleNotFoundError('peft is required for LoRA tests.')
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            target_modules=['q_proj', 'v_proj']
        )
        profiler.peftModel(lora_config)
    print(f'------------------{profiler.model_id} Profile flops------------------')
    profiler.get_calflops(bs, seq, device)

def test_memory(path = '/workspace/code/HF-LLM-Profiler/models/Meta-Llama-3.1-70B', bs = 8, seq = 512, device = 'cuda:0', test_lora = False):
    '''
        Test the memory of the model.
    '''
    profiler = ModelProfiler(path, verbose=True)

    if test_lora:
        if LoraConfig is None:
            raise ModuleNotFoundError('peft is required for LoRA tests.')
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            target_modules=['q_proj', 'v_proj']
        )
        profiler.peftModel(lora_config)

    print(f'------------------{profiler.model_id} Profile memory------------------')
    # Test forward memory
    clear_device_cache(device)
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, True, 'memory', 5, 10, False, False))
    thread.start()
    thread.join()
    # Test backward memory
    clear_device_cache(device)
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, False, 'memory', 5, 10, False, False))
    thread.start()
    thread.join()

def test_time(path = '/workspace/code/HF-LLM-Profiler/models/Meta-Llama-3.1-70B', bs = 8, seq = 512, device = 'cuda:0', test_lora = False):
    '''
        Test the speed of the model.
    '''
    profiler = ModelProfiler(path, verbose=True)
    if test_lora:
        if LoraConfig is None:
            raise ModuleNotFoundError('peft is required for LoRA tests.')
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            target_modules=['q_proj', 'v_proj']
        )
        profiler.peftModel(lora_config)
    print(f'------------------{profiler.model_id} Profile time------------------')
    # Test forward time
    clear_device_cache(device)
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, True, 'time', 5, 10, False, False))
    thread.start()
    thread.join()
    # Test backward time
    clear_device_cache(device)
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, False, 'time', 5, 10, False, False))
    thread.start()
    thread.join()

def test():
    '''
        Test the Profiler's method.
        Note:
            Profile gpu memory/time twice must using subprocess to run the script.
    '''
    path, device, test_lora = '/workspace/code/HF-LLM-Profiler/models/Qwen2.5-0.5B', "cuda:5", True
    print_model(path=path)
    test_flops(path=path, device=device, test_lora=test_lora)
    test_memory(path=path, device=device, test_lora=test_lora)
    test_time(path=path, device=device, test_lora=test_lora)

def test_components(path='/data/HF_MODELS/Qwen2.5-3B', bs=1, seq=128, device='cuda:0', dtype=torch.float16):
    '''
        Test embedding forward time and FFN forward/backward time-memory.
    '''
    print(f'------------------{path.split("/")[-1]} Component Profile------------------')

    embed_profiler = EmbeddingProfiler(path, verbose=True, dtype=dtype, device=device)
    clear_device_cache(device)
    thread = multiprocessing.Process(target=embed_profiler.profile, args=(bs, seq, device, True, 'time', 3, 5))
    thread.start()
    thread.join()

    ffn_profiler = FFNProfiler(path, verbose=True, dtype=dtype, device=device)
    for fwd_flag, flag_name in [(True, 'forward'), (False, 'forward+backward')]:
        clear_device_cache(device)
        thread = multiprocessing.Process(target=ffn_profiler.profile, args=(bs, seq, device, fwd_flag, 'time', 3, 5, False, True))
        thread.start()
        thread.join()
        clear_device_cache(device)
        thread = multiprocessing.Process(target=ffn_profiler.profile, args=(bs, seq, device, fwd_flag, 'memory', 3, 5, False, True))
        thread.start()
        thread.join()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    if len(sys.argv) > 1 and sys.argv[1] == 'components':
        test_components()
    else:
        test()
