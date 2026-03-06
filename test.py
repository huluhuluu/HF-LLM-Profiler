import torch
import multiprocessing
from Profiler import Profiler, ModelProfiler
from peft import LoraConfig

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
    torch.cuda.empty_cache()
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, True, 'memory', 5, 10, False, False))
    thread.start()
    thread.join()
    # Test backward memory
    torch.cuda.empty_cache()
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, False, 'memory', 5, 10, False, False))
    thread.start()
    thread.join()

def test_time(path = '/workspace/code/HF-LLM-Profiler/models/Meta-Llama-3.1-70B', bs = 8, seq = 512, device = 'cuda:0', test_lora = False):
    '''
        Test the speed of the model.
    '''
    profiler = ModelProfiler(path, verbose=True)
    if test_lora:
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
    torch.cuda.empty_cache()
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, True, 'time', 5, 10, False, False))
    thread.start()
    thread.join()
    # Test backward time
    torch.cuda.empty_cache()
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

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    test()