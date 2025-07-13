import time
import torch
import multiprocessing
from flops_counter import calculate_flops
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel

# global variables
# used for different model's config
MODEL_LAYER_KEY = {'default': 'num_hidden_layers'}
MODEL_HIDEEN_SIZE_KEY = {'default': 'hidden_size'}
MODEL_DTYPE_KEY = {'default': 'torch_dtype'}
MODEL_EMBED_KEY = {'default': 'embed_tokens'}
MODEL_ROTATE_KEY = {'default': 'rotary_emb'}
MODEL_TRANS_KEY = {'default': 'layers'}

class Profiler(object):
    '''
        This is a base class for the profiler.
    '''
    from abc import ABC, abstractmethod
    def __init__(self, model_id_or_path: str, verbose: bool = False, dtype:torch.dtype = None):
        self.model_id_or_path = model_id_or_path 
        self.verbose = verbose
        self.model_id = self.get_model_id(model_id_or_path)
        # get config and empty model
        self.config = AutoConfig.from_pretrained(model_id_or_path)
        if dtype is not None:
            self.config.torch_dtype = dtype
        with init_empty_weights():
            self.model = AutoModel.from_config(self.config)
        # get layers, hidden size and tensor dtype info
        self.layer, self.hidden_size, self.dtype_ = self.get_model_info(self.config, self.model_id) # transformer block layers and hidden size

    def get_model_id(self, model_id_or_path: str = None):
        '''
            Get the model id from the model path or hf hub name.
            If the model path is a local path, return the last part of the path.
            If the model path is a huggingface model id, return the model id.

            eg: '/data/HUGGINGFACE/Falcon3-10B-Base' -> 'Falcon3-10B-Base'
            eg: 'meta-llama/Llama-2-7b-chat-hf' -> 'Llama-2-7b-chat-hf'
        '''
        if '/' in model_id_or_path:
            return model_id_or_path.split('/')[-1]
        return model_id_or_path
    
    def get_attr(self, key_dict, class_, model_id: str):
        '''
            Get the class_'s attribute by the model id.
            If the model id is not in the key_dict, use the default key.
            If the model id is in the key_dict, use the model id as the key.
        '''
        try:
            key = key_dict.get(model_id, key_dict['default'])
            if class_ is None:
                value = key_dict.get(model_id, key_dict['default'])
            else:
                value = getattr(class_, key, None)
            if value is None:
                raise ValueError(f"Model {model_id} Attr {key} | not supported, please check the model id or path.")
            return value
        except:
            raise ValueError(f"Model {model_id} not supported, please check the model id or path.")

    def get_model_info(self, config, model_id: str):
        '''
            Get the number of hidden layers and hidden size and tensor_dtype of the model.
        '''
        global MODEL_LAYER_KEY, MODEL_HIDEEN_SIZE_KEY, MODEL_DTYPE_KEY

        # get the number of hidden layers and hidden size from the config
        return  self.get_attr(MODEL_LAYER_KEY, config, model_id), \
                self.get_attr(MODEL_HIDEEN_SIZE_KEY, config, model_id), \
                self.get_attr(MODEL_DTYPE_KEY, config, model_id)

    @classmethod
    def count_param(cls, model):
        '''
            Count the number of parameters and trainable parameters in the model both metadata and actual data form.
        '''
        meta_total_params = sum(p.numel() for p in model.parameters() if p.device == torch.device('meta'))
        meta_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad and p.device == torch.device('meta'))
        actual_total_params = sum(p.numel() for p in model.parameters())
        actual_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if cls.verbose:
            print(f'Trainable params(actual|meta): {actual_trainable_params / 1e9:.2f}B|{meta_trainable_params / 1e9:.2f}B')
            print(f'Total params(actual|meta): {actual_total_params / 1e9:.2f}B|{meta_total_params / 1e9:.2f}B')
        return actual_trainable_params, meta_trainable_params, actual_total_params, meta_total_params
     
    def mask(self, mask_):
        '''
            Generate the attention mask for the model.
        '''
        if mask_.dim() == 3:
            mask_ = mask_[:, None, :, :]
        elif mask_.dim() == 2:
            mask_ = mask_[:, None, None, :]
        mask_ = (1.0 - mask_.to(self.dtype_)) * torch.finfo(self.dtype_).min
        return mask_
    
    @abstractmethod
    def get_model(self, model):
        '''
            Get the empty model part needed from the whole model(self.model).
        '''
        ...
    
    @abstractmethod
    def init_empty_model(self, device = 'cpu'):
        '''
            Initialize the empty model we got with metedata form.
        '''
        ...

    @abstractmethod
    def gen_input(self, bs: int = 8, seq_len: int = 512, device = 'cpu'):
        '''
            Generate the input for the model.
        '''
        ...

    @abstractmethod
    def profile(self, bs: int, seq_len: int, device = 'cpu', fwd_flag = True, profile_flag: str = 'time',
                        skip_round: int = 100, test_round : int = 500):
        '''
            Profile the forward or backward pass of the model.
            This is a placeholder function and should be implemented in the future.

            args:
                bs: batch size
                seq_len: sequence length
                device: device to run the model on
                fwd_flag: True for forward pass, False for backward pass
                profile_time_flag: 'time' for profiling time, 'memory' for profiling memory
                skip_round: number of rounds to skip for profiling
                test_round: number of rounds to test for profiling
        '''
        ...
    
class ModelProfiler(Profiler):
    '''
        This class is used to estimate the total GPU memory, runtime, and FLOPs 
        for the full model by **scaling up the results from one block**. 
    '''
    def __init__(self, model_id_or_path: str, verbose: bool = False, dtype:torch.dtype = None):
        super().__init__(model_id_or_path, verbose=verbose, dtype=dtype)
        self.trans, self.embeds, self.rotate = self.get_model(self.model) # get the model from the config

    def get_model(self, model):
        '''
            Get the empty model from the config.
            Get the transformer/embeds/rotate layer from the empty model.
            Args:
                model: the empty model from the config.
            Return:
                transformer layer, embedding layer, rotation layer
        '''
        global MODEL_EMBED_KEY, MODEL_ROTATE_KEY, MODEL_TRANS_KEY
        # get only a singel layer of the transformer
        return  self.get_attr(MODEL_TRANS_KEY, model, self.model_id)[1], \
                self.get_attr(MODEL_EMBED_KEY, model, self.model_id), \
                self.get_attr(MODEL_ROTATE_KEY, model, self.model_id)

    def  init_lora_model(self, rank:int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05, target_modules: list = None):
        '''
            Initialize the model with lora method.
            Args:
                rank: the rank of the lora
                lora_alpha: the alpha of the lora
                lora_dropout: the dropout of the lora
                lora_method: the method of the lora, default is 'lora'
        '''
        # if target_modules is None: use the default target modules
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias='none',
            target_modules=target_modules if target_modules is not None else ['q_proj', 'v_proj']
        )

        try:
            self.trans = get_peft_model(self.trans, lora_config)
        except:
            raise ValueError(f"target_modules {target_modules} errors, please check the model struct:\n {self.trans}")

    def init_empty_model(self, device = 'cpu'):
        '''
            Initialize the empty model we got with metedata form.
            Note:
                To test gpu memory usage, get_calflops() and profile_forward() or profile_backward()
                should not be called in the same time.
        '''
        self.trans = self.trans.to_empty(device=device)
        self.embeds = self.embeds.to_empty(device=device)
        self.rotate = self.rotate.to_empty(device=device)
    
    def gen_input(self, bs: int = 8, seq_len: int = 512, device = 'cpu'):
        '''
            Generate the input for the model.
            The input is a tensor of shape (bs, seq_len) with random integers.
        '''
        # input args' shape
        input_ids_shape = (bs, seq_len)
        input_shape, attention_shape  = [bs, seq_len, self.hidden_size], [bs, seq_len]

        # generate the input
        input_ids = torch.ones(input_ids_shape, dtype=torch.int64).to(device)
        position_ids = torch.arange(0, input_shape[1], dtype=torch.int64).unsqueeze(0).to(device)
        attention_mask = torch.ones(attention_shape, dtype=torch.int64)
        # mask the attention mask
        attention_mask = self.mask(attention_mask).to(device)
        
        # generate the input(hidden state, embeds)
        hidden_state = self.embeds(input_ids)        
        input_embeds = self.rotate(hidden_state, position_ids)
        return input_embeds, attention_mask, hidden_state

    def get_calflops(self, bs: int = 8, seq_len: int = 512, device = 'cpu'):
        '''
            Get the calflops of the model by scaling up the results from one block. 

            args:
                bs: batch size
                seq_len: sequence length
                device: device to run the model on

            return:
                fwd_flops: forward flops
                bwd_flops: backward flops
                param: params
        '''
        # gen input for the model transformer
        self.init_empty_model(device = device)
        input_embeds, attention_mask, hidden_state = self.gen_input(bs, seq_len, device)
        kwargs = {
            'position_embeddings': input_embeds,
            'attention_mask': attention_mask,
            'hidden_states': hidden_state
        }
        
        # get the flops of the model in forward and backward pass with a single transformer block
        fwd_trans_flops, fwd_trans_macs, trans_params = calculate_flops(  
                                                    model = self.trans,
                                                    kwargs = kwargs,
                                                    include_backPropagation=False,
                                                    print_results = False
                                                )
        bwd_trans_flops, bwd_trans_macs, trans_params = calculate_flops(  
                                                    model = self.trans,
                                                    kwargs = kwargs,
                                                    include_backPropagation=True,
                                                    print_results = False
                                                )
        def split_res(res: str):
            '''
                Get the number from the string.
                eg: '631.41 GFLOPS' -> (631.41, 'GFLOPS')
                eg: '315.68 GMACs' -> (315.68, 'GMACs')
                eg: '77.08 M' -> (77.08, 'M')
            '''
            res = res.split(' ')
            if len(res) == 2:
                return float(res[0]), res[1]
            raise ValueError(f"Invalid result format: {res}")
        
        def scale_up(res: str, layer: int):
            '''
                Scale up the result by the number of layers.
                eg: '631.41 GFLOPS' -> '631.41 * 12 GFLOPS' # 12 is the number of layers
                eg: '315.68 GMACs' -> '315.68 * 12 GMACs'   # 12 is the number of layers
                eg: '77.08 M' -> '77.08 * 12 M'             # 12 is the number of layers
            '''
            res, res_units = split_res(res)
            res = res * layer
            return f'{res:.2f} {res_units}'
        
        if self.verbose:
            print(f'{self.model_id} | batch size {bs} | seq_len {seq_len} | layers {self.layer} | {self.dtype_}')
        
        # scale up the result by the number of layers
        fwd_flops, bwd_flops, param = scale_up(fwd_trans_flops, self.layer), \
                                      scale_up(bwd_trans_flops, self.layer), \
                                      scale_up(trans_params, self.layer)
        
        if self.verbose:
            print(f'forward flops: {fwd_flops} | backward flops: {bwd_flops} | params: {param}')
        return fwd_flops, bwd_flops, param

    def profile(self, bs: int, seq_len: int, device = 'cpu', fwd_flag = True, profile_flag: str = 'time',
                        skip_round: int = 100, test_round : int = 500):
        '''
            Profile the forward or backward pass of the model.
            This is a placeholder function and should be implemented in the future.

            args:
                bs: batch size
                seq_len: sequence length
                device: device to run the model on
                fwd_flag: True for forward pass, False for forward + backward pass
                profile_time_flag: 'time' for profiling time, 'memory' for profiling memory
                skip_round: number of rounds to skip for profiling
                test_round: number of rounds to test for profiling

            return:
                (end_time - begin_time) / test_round: average time for single attention block
                (model_memory - begin_memory) * self.layer / 1024**3: model memory
                (end_memory - model_memory) * self.layer / 1024**3: activation memory
                (end_memory - begin_memory) * self.layer / 1024**3: total memory
        '''
        # gen input for the model transformer
        # for profiler gpu memory set model init device to 'cpu'
        self.init_empty_model(device = 'cpu')
        input_embeds, attention_mask, hidden_state = self.gen_input(bs, seq_len, 'cpu')
        kwargs = {
            'position_embeddings': (input_embeds[0].detach().to(device), input_embeds[1].detach().to(device)),
            'attention_mask': attention_mask.detach().to(device),
            'hidden_states': hidden_state.detach().to(device) if fwd_flag else hidden_state.detach().to(device).requires_grad_(True)
        }
        # set torch device
        torch.cuda.set_device(device)
        # set params
        begin_time, end_time = 0, 0
        begin_memory, model_memory, end_memory = torch.cuda.max_memory_allocated(device), 0, 0
        
        # get the model's memory usage
        self.trans.to(device)
        model_memory = torch.cuda.max_memory_allocated(device)

        if fwd_flag:
            # forward pass
            with torch.no_grad():
                for _ in range(test_round + skip_round * 2):
                    if _ == skip_round:
                        begin_time = time.time()
                    elif _ == skip_round + test_round:
                        end_time = time.time()
                        end_memory = torch.cuda.max_memory_allocated(device)
                    if profile_flag == 'time':
                        kwargs = {
                            'position_embeddings': (input_embeds[0].detach().to(device), input_embeds[1].detach().to(device)),
                            'attention_mask': attention_mask.detach().to(device),
                            'hidden_states': hidden_state.detach().to(device) if fwd_flag else hidden_state.detach().to(device).requires_grad_(True)
                        }
                    # forward
                    tensor = self.trans(**kwargs)
        else:
            # backward pass
            for _ in range(test_round + skip_round * 2):
                if _ == skip_round:
                    begin_time = time.time()
                elif _ == skip_round + test_round:
                    end_time = time.time()
                    end_memory = torch.cuda.max_memory_allocated(device)
                if profile_flag == 'time':
                    kwargs = {
                        'position_embeddings': (input_embeds[0].detach().to(device), input_embeds[1].detach().to(device)),
                        'attention_mask': attention_mask.detach().to(device),
                        'hidden_states': hidden_state.detach().to(device) if fwd_flag else hidden_state.detach().to(device).requires_grad_(True)
                    }
                self.trans.zero_grad()
                # forward
                tensor = self.trans(**kwargs)
                # backward
                torch.autograd.backward(tensor[0] if type(tensor) is tuple else tensor, grad_tensors=torch.ones_like(tensor[0] if type(tensor) is tuple else tensor))

        # print the result
        if self.verbose:
            print(f'{self.model_id} | batch size {bs} | seq_len {seq_len} | layers {self.layer} | {self.dtype_} | {'forward' if fwd_flag else 'backward'}')
            if profile_flag == 'time':
                print(f'block runing time: {(end_time - begin_time) / test_round:.5f} s')
            else:
                print(f'model memory: {(model_memory - begin_memory) / 1024**3:.4f}/{(model_memory - begin_memory) * self.layer / 1024**3:.4f} GB')
                print(f'activation memory: {(end_memory - model_memory) / 1024**3:.4f}/{(end_memory - model_memory) * self.layer / 1024**3:.4f} GB')
                print(f'total memory: {(end_memory - begin_memory) / 1024**3:.4f}/{(end_memory - begin_memory) * self.layer / 1024**3:.4f} GB')
        return (end_time - begin_time) / test_round, \
                (model_memory - begin_memory) * self.layer / 1024**3, \
                (end_memory - model_memory) * self.layer / 1024**3, \
                (end_memory - begin_memory) * self.layer / 1024**3

class EmbeddingProfiler(Profiler):
    '''
        This class is used to estimate the total GPU memory, runtime, and FLOPs 
        for the embedding model. 
    '''
    pass

class FFNProfiler(Profiler):
    '''
        This class is used to estimate the total GPU memory, runtime, and FLOPs 
        for the model's FFN(at the last layers of the model).
    '''
    pass

def test():
    '''
        Test the ProfileModel class's method.
    '''
    bs, seq, device, rank, lora_alpha, lora_dropout = 8, 512, 'cuda:0', 8, 16, 0.05
    path = '/data/HF_MODELS/Qwen2.5-3B' # Qwen2.5-3B Falcon3-10B-Base
    profiler = ModelProfiler(path, dtype=torch.float32, verbose=True)
    profiler.init_lora_model(
        rank=rank, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout, 
        target_modules=['q_proj', 'v_proj']
    )
    # print(profiler.trans)
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

if __name__ == '__main__':
    test()