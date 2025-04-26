import torch
import multiprocessing
from Profiler import Profiler, ModelProfiler

def print_model(path = '/docker/data/HUGGINGFACE/Llama-3.1-70B-Instruct'):
    '''
        Print the model's structure w/o donwload hole model.
    '''
    profiler = Profiler(path, verbose=True)
    print(f'------------------{profiler.model_id}------------------')
    print(profiler.model)

def test_flops(path = '/docker/data/HUGGINGFACE/Llama-3.1-70B-Instruct', bs = 8, seq = 512, device = 'cuda:0'):
    '''
        Test the flops of the model.
    '''
    profiler = ModelProfiler(path, verbose=True)
    print(f'------------------{profiler.model_id} Profile flops------------------')
    profiler.get_calflops(bs, seq, device)

def test_memory(path = '/docker/data/HUGGINGFACE/Llama-3.1-70B-Instruct', bs = 8, seq = 512, device = 'cuda:0'):
    '''
        Test the memory of the model.
    '''
    profiler = ModelProfiler(path, verbose=True)
    print(f'------------------{profiler.model_id} Profile memory------------------')
    # Test forward memory
    torch.cuda.empty_cache()
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, True, 'memory', 50, 100))
    thread.start()
    thread.join()
    # Test backward memory
    torch.cuda.empty_cache()
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, False, 'memory', 50, 100))
    thread.start()
    thread.join()

def test_time(path = '/docker/data/HUGGINGFACE/Llama-3.1-70B-Instruct', bs = 8, seq = 512, device = 'cuda:0'):
    '''
        Test the speed of the model.
    '''
    profiler = ModelProfiler(path, verbose=True)
    print(f'------------------{profiler.model_id} Profile time------------------')
    # Test forward time
    torch.cuda.empty_cache()
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, True, 'time', 50, 100))
    thread.start()
    thread.join()
    # Test backward time
    torch.cuda.empty_cache()
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, False, 'time', 50, 100))
    thread.start()
    thread.join()

def test():
    '''
        Test the Profiler's method.
        Note:
            Profile gpu memory/time twice must using subprocess to run the script.
    '''
    print_model()
    test_flops()
    test_memory()
    test_time()

    # TODO: test getattr recursive .

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    test()