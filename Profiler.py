import time
import torch
import multiprocessing
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel
from model_config import (
    MODEL_LAYER_KEY,
    MODEL_HIDEEN_SIZE_KEY,
    MODEL_DTYPE_KEY,
    MODEL_EMBED_KEY,
    MODEL_ROTATE_KEY,
    MODEL_TRANS_KEY,
)

try:
    from flops_counter import calculate_flops
except ModuleNotFoundError:
    calculate_flops = None

try:
    from peft import LoraConfig, PeftConfig, get_peft_model
except ModuleNotFoundError:
    LoraConfig = None
    PeftConfig = object
    get_peft_model = None

class Profiler(object):
    '''
        This is a base class for the profiler.
    '''
    from abc import ABC, abstractmethod
    def __init__(self, model_id_or_path: str, verbose: bool = False, dtype:torch.dtype = None, device: str = 'cpu'):
        self.model_id_or_path = model_id_or_path 
        self.verbose = verbose
        self.model_id = self.get_model_id(model_id_or_path)
        # get config and empty model
        self.config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
        # Use new 'dtype' attribute instead of deprecated 'torch_dtype'
        if dtype is not None:
            self.config.dtype = dtype
        with init_empty_weights():
            self.model = AutoModel.from_config(self.config, trust_remote_code=True)
        self.model_type = getattr(self.config, 'model_type', self.model_id)
        # get layers, hidden size and tensor dtype info
        self.layer, self.hidden_size, self.dtype_ = self.get_model_info(self.config, self.model_type)
        self.backend = self._detect_backend(device)

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
                value = class_
                for sub_key in key.split('.'):
                    value = getattr(value, sub_key, None)
                    if value is None:
                        break
            if value is None:
                raise ValueError(f"Model {model_id} Attr {key} | not supported, please check the model id or path.")
            return value
        except:
            raise ValueError(f"Model {model_id} not supported, please check the model id or path.")

    def get_model_info(self, config, model_id: str):
        '''
            Get the number of hidden layers and hidden size and tensor_dtype of the model.
        '''
        # get the number of hidden layers and hidden size from the config
        return  self.get_attr(MODEL_LAYER_KEY, config, model_id), \
                self.get_attr(MODEL_HIDEEN_SIZE_KEY, config, model_id), \
                self.get_attr(MODEL_DTYPE_KEY, config, model_id)

    @staticmethod
    def _detect_backend(device: str):
        if 'cuda' in device:
            return 'cuda'
        if 'npu' in device:
            return 'npu'
        return 'cpu'

    @classmethod
    def _device_module(cls, device: str):
        backend = cls._detect_backend(device)
        if backend == 'cpu':
            return None
        return getattr(torch, backend, None)

    @classmethod
    def _set_device(cls, device: str):
        module = cls._device_module(device)
        if module is not None and hasattr(module, 'set_device'):
            module.set_device(device)

    @classmethod
    def _synchronize(cls, device: str):
        module = cls._device_module(device)
        if module is not None and hasattr(module, 'synchronize'):
            module.synchronize(device)

    @classmethod
    def _empty_cache(cls, device: str):
        module = cls._device_module(device)
        if module is not None and hasattr(module, 'empty_cache'):
            module.empty_cache()

    @classmethod
    def _reset_peak_memory_stats(cls, device: str):
        module = cls._device_module(device)
        if module is not None and hasattr(module, 'reset_peak_memory_stats'):
            module.reset_peak_memory_stats(device)

    @classmethod
    def _max_memory_allocated(cls, device: str):
        module = cls._device_module(device)
        if module is None or not hasattr(module, 'max_memory_allocated'):
            return 0
        return module.max_memory_allocated(device)

    @classmethod
    def _memory_allocated(cls, device: str):
        module = cls._device_module(device)
        if module is None or not hasattr(module, 'memory_allocated'):
            return 0
        return module.memory_allocated(device)

    @staticmethod
    def _parameter_memory_bytes(module):
        return sum(param.numel() * param.element_size() for param in module.parameters() if param is not None)

    @staticmethod
    def _primary_tensor(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output:
            return Profiler._primary_tensor(output[0])
        if isinstance(output, dict):
            for value in output.values():
                tensor = Profiler._primary_tensor(value)
                if tensor is not None:
                    return tensor
        return None

    @classmethod
    def _detach_to_device(cls, value, device: str, requires_grad: bool = False):
        if torch.is_tensor(value):
            tensor = cls.to_device(value.detach(), device)
            if requires_grad:
                tensor = tensor.requires_grad_(True)
            return tensor
        if isinstance(value, tuple):
            return tuple(cls._detach_to_device(item, device, requires_grad=False) for item in value)
        if isinstance(value, list):
            return [cls._detach_to_device(item, device, requires_grad=False) for item in value]
        return value

    @abstractmethod
    def peftModel(self, config: PeftConfig = None):
        '''
            Get the peft model from the config.
        '''
        ... 

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
            NPU FlashAttention supports: [B, N, Sq, Skv], [B, 1, Sq, Skv], [1, 1, Sq, Skv] and [Sq, Skv]
            Generate [1, 1, Sq, Skv] shape for NPU compatibility.
        '''
        if mask_.dim() == 2:
            # Input shape: [bs, seq_len]
            # For NPU compatibility, use [1, 1, seq_len, seq_len] (broadcastable to all batches)
            seq_len = mask_.shape[1]
            # Create a causal mask or full mask of shape [seq_len, seq_len]
            mask_2d = torch.ones((seq_len, seq_len), dtype=mask_.dtype)
            mask_ = mask_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        elif mask_.dim() == 3:
            mask_ = mask_[:, None, :, :]
        mask_ = (1.0 - mask_.to(self.dtype_)) * torch.finfo(self.dtype_).min
        return mask_

    @classmethod
    def to_device(cls, obj, device):
        if device == 'cpu':
            return obj.to(device)
        if 'npu' in device:
            return obj.npu()
        return obj.to(device)

    def _materialize_module(self, module, device):
        '''
            Materialize a module with meta tensors to actual tensors on the target device.
        '''
        module = module.to_empty(device=device)

        def reset_module_params(mod):
            for _, child in mod.named_children():
                reset_module_params(child)
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()

        reset_module_params(module)
        return module

    def _profile_module(self, module, input_factory, device='cpu', fwd_flag=True, profile_flag='time',
                        skip_round: int = 10, test_round: int = 5, scale_factor: int = 1):
        self.backend = self._detect_backend(device)
        self._set_device(device)
        self._empty_cache(device)
        self._synchronize(device)
        self._reset_peak_memory_stats(device)

        begin_time, end_time = 0.0, 0.0
        begin_memory = self._memory_allocated(device)
        self.__class__.to_device(module, device)
        self._synchronize(device)
        model_memory = self._memory_allocated(device)
        self._reset_peak_memory_stats(device)
        end_memory = model_memory

        total_round = test_round + skip_round * 2
        for idx in range(total_round):
            if idx == skip_round:
                self._synchronize(device)
                begin_time = time.time()

            args, kwargs = input_factory(device, fwd_flag)
            if hasattr(module, 'zero_grad'):
                module.zero_grad(set_to_none=True)

            if fwd_flag:
                with torch.no_grad():
                    output = module(*args, **kwargs)
            else:
                output = module(*args, **kwargs)
                loss = self._primary_tensor(output)
                if loss is None:
                    raise ValueError('Profiler output does not contain a tensor for backward.')
                torch.autograd.backward(loss, grad_tensors=torch.ones_like(loss))

            self._synchronize(device)
            if idx == skip_round + test_round - 1:
                end_time = time.time()
                end_memory = self._max_memory_allocated(device)

        model_delta = max(model_memory - begin_memory, 0)
        act_delta = max(end_memory - model_memory, 0)
        total_delta = max(end_memory - begin_memory, 0)

        return (end_time - begin_time) / test_round * scale_factor, \
               model_delta * scale_factor / 1024**3, \
               act_delta * scale_factor / 1024**3, \
               total_delta * scale_factor / 1024**3

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
                        skip_round: int = 10, test_round : int = 5, skip_init: bool = False, block_flag: bool = True):
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
    def __init__(self, model_id_or_path: str, verbose: bool = False, dtype:torch.dtype = None, device: str = 'cpu'):
        super().__init__(model_id_or_path, verbose=verbose, dtype=dtype, device=device)
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
        # get only a single layer of the transformer
        return  self.get_attr(MODEL_TRANS_KEY, model, self.model_type)[0], \
                self.get_attr(MODEL_EMBED_KEY, model, self.model_type), \
                self.get_attr(MODEL_ROTATE_KEY, model, self.model_type)

    def init_empty_model(self, device = 'cpu'):
        '''
            Initialize the empty model we got with metedata form.
            Note:
                To test gpu memory usage, get_calflops() and profile_forward() or profile_backward()
                should not be called in the same time.
        '''
        # Materialize meta tensors to actual tensors with physical memory
        # to_empty() creates meta tensors without physical memory, which causes errors
        # when trying to run actual computations (e.g., embedding lookup)
        self.trans = self._materialize_module(self.trans, device)
        self.embeds = self._materialize_module(self.embeds, device)
        self.rotate = self._materialize_module(self.rotate, device)
    
    def gen_input(self, bs: int = 8, seq_len: int = 512, device = 'cpu'):
        '''
            Generate the input for the model.
            The input is a tensor of shape (bs, seq_len) with random integers.
        '''
        # input args' shape
        input_ids_shape = (bs, seq_len)
        attention_shape  = [bs, seq_len]

        # generate the input
        input_ids = self.__class__.to_device(torch.ones(input_ids_shape, dtype=torch.int64), device)
        position_ids = torch.arange(0, seq_len, dtype=torch.int64).unsqueeze(0).expand(bs, -1)
        attention_mask = torch.ones(attention_shape, dtype=torch.int64)

        # move embeds and rotate to target device (self.mask is a method, not a module)
        self.embeds = self.__class__.to_device(self.embeds, device)
        hidden_state = self.embeds(input_ids)
        attention_mask = self.__class__.to_device(self.mask(attention_mask), device)

        if self.model_type == 'gpt2':
            self.rotate = self.__class__.to_device(self.rotate, device)
            position_embeds = self.rotate(self.__class__.to_device(position_ids, device))
            hidden_state = hidden_state + position_embeds
            return {
                'hidden_states': hidden_state,
                'attention_mask': attention_mask,
            }

        self.rotate = self.__class__.to_device(self.rotate, device)
        position_ids = self.__class__.to_device(position_ids, device)
        input_embeds = self.rotate(hidden_state, position_ids)
        return {
            'position_embeddings': input_embeds,
            'attention_mask': attention_mask,
            'hidden_states': hidden_state,
        }

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
        if calculate_flops is None:
            raise ModuleNotFoundError('calflops is required for FLOPs profiling. Install dependencies from HF-LLM-Profiler/requirements.txt.')
        # gen input for the model transformer
        self.init_empty_model(device = device)
        kwargs = self.gen_input(bs, seq_len, device)
        
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
                        skip_round: int = 10, test_round : int = 5, skip_init: bool = False, 
                        block_flag: bool = True):
        '''
            Profile the forward or backward pass of the model.

            args:
                bs: batch size
                seq_len: sequence length
                device: device to run the model on
                fwd_flag: True for forward pass, False for forward + backward pass
                profile_flag: 'time' for profiling time, 'memory' for profiling memory
                skip_round: number of rounds to skip for profiling
                test_round: number of rounds to test for profiling
                skip_init: if True, skip model initialization (use when model already initialized, e.g., after LoRA)
                block_flag: if True, profile only one block, else profile the whole model by scaling up the results from one block
            return:
                (end_time - begin_time) / test_round: average time for single attention block
                (model_memory - begin_memory) * self.layer / 1024**3: model memory
                (end_memory - model_memory) * self.layer / 1024**3: activation memory
                (end_memory - begin_memory) * self.layer / 1024**3: total memory
        '''
        # gen input for the model transformer
        # for profiler gpu memory set model init device to 'cpu'
        if not skip_init:
            self.init_empty_model(device = 'cpu')
        base_kwargs = self.gen_input(bs, seq_len, 'cpu' if not skip_init else device)

        def input_factory(target_device, is_forward):
            kwargs = {}
            for key, value in base_kwargs.items():
                kwargs[key] = self._detach_to_device(value, target_device, requires_grad=(key == 'hidden_states' and not is_forward))
            return (), kwargs

        run_time, model_mem, act_mem, total_mem = self._profile_module(
            self.trans,
            input_factory=input_factory,
            device=device,
            fwd_flag=fwd_flag,
            profile_flag=profile_flag,
            skip_round=skip_round,
            test_round=test_round,
            scale_factor=1 if block_flag else self.layer,
        )

        # print the result
        if self.verbose:
            print(f"{self.model_id} | batch size {bs} | seq_len {seq_len} | layers {self.layer} | {self.dtype_} | {'forward' if fwd_flag else 'backward'}")
            if profile_flag == 'time':
                if block_flag:
                    print(f'block runing time: {run_time:.5f} s')
                else:
                    print(f'model runing time: {run_time:.5f} s')
            else:
                if block_flag:
                    print(f'block model memory: {model_mem:.4f} GB')
                    print(f'block activation memory: {act_mem:.4f} GB')
                    print(f'block total memory: {total_mem:.4f} GB')
                else:
                    print(f'model memory: {model_mem / self.layer:.4f}/{model_mem:.4f} GB')
                    print(f'activation memory: {act_mem / self.layer:.4f}/{act_mem:.4f} GB')
                    print(f'total memory: {total_mem / self.layer:.4f}/{total_mem:.4f} GB')
        return run_time, model_mem, act_mem, total_mem

    def peftModel(self, config: PeftConfig = None):
        if LoraConfig is None or get_peft_model is None:
            raise ModuleNotFoundError('peft is required for LoRA profiling. Install dependencies from HF-LLM-Profiler/requirements.txt.')
        if config is None:
            rank, lora_alpha, lora_dropout = 8, 16, 0.05
            config = LoraConfig(
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias='none',
                target_modules=['q_proj', 'v_proj']
            )
        try:
            self.trans = get_peft_model(self.trans, config)
        except:
            raise ValueError(f"Get peft model failed, please check the config:\n {config}")
    
class EmbeddingProfiler(Profiler):
    '''
        This class is used to estimate the total GPU memory, runtime, and FLOPs 
        for the embedding model. 
    '''
    def __init__(self, model_id_or_path: str, verbose: bool = False, dtype: torch.dtype = None, device: str = 'cpu'):
        super().__init__(model_id_or_path, verbose=verbose, dtype=dtype, device=device)
        self.embeds = self.get_model(self.model)

    def get_model(self, model):
        return self.get_attr(MODEL_EMBED_KEY, model, self.model_type)

    def init_empty_model(self, device='cpu'):
        self.embeds = self._materialize_module(self.embeds, device)

    def gen_input(self, bs: int = 8, seq_len: int = 512, device='cpu'):
        return self.__class__.to_device(torch.ones((bs, seq_len), dtype=torch.int64), device)

    def profile(self, bs: int, seq_len: int, device='cpu', fwd_flag=True, profile_flag: str = 'time',
                skip_round: int = 10, test_round: int = 5, skip_init: bool = False, block_flag: bool = True):
        if not skip_init:
            self.init_empty_model(device='cpu')
        input_ids = self.gen_input(bs, seq_len, 'cpu' if not skip_init else device)

        def input_factory(target_device, _):
            return (self._detach_to_device(input_ids, target_device),), {}

        run_time, model_mem, act_mem, total_mem = self._profile_module(
            self.embeds,
            input_factory=input_factory,
            device=device,
            fwd_flag=fwd_flag,
            profile_flag=profile_flag,
            skip_round=skip_round,
            test_round=test_round,
            scale_factor=1,
        )

        if self.verbose:
            print(f"{self.model_id} | embedding | batch size {bs} | seq_len {seq_len} | {self.dtype_} | {'forward' if fwd_flag else 'backward'}")
            if profile_flag == 'time':
                print(f'embedding runing time: {run_time:.5f} s')
            else:
                print(f'embedding model memory: {model_mem:.4f} GB')
                print(f'embedding activation memory: {act_mem:.4f} GB')
                print(f'embedding total memory: {total_mem:.4f} GB')
        return run_time, model_mem, act_mem, total_mem

    def peftModel(self, config: PeftConfig = None):
        raise NotImplementedError('EmbeddingProfiler does not support PEFT profiling.')

class FFNProfiler(Profiler):
    '''
        This class is used to estimate the total GPU memory, runtime, and FLOPs 
        for the model's FFN(at the last layers of the model).
    '''
    def __init__(self, model_id_or_path: str, verbose: bool = False, dtype: torch.dtype = None, device: str = 'cpu'):
        super().__init__(model_id_or_path, verbose=verbose, dtype=dtype, device=device)
        self.ffn = self.get_model(self.model)

    def get_model(self, model):
        layer = self.get_attr(MODEL_TRANS_KEY, model, self.model_type)[0]
        ffn = getattr(layer, 'mlp', None)
        if ffn is None:
            raise ValueError(f'Model {self.model_id} does not expose an `mlp` module for FFN profiling.')
        return ffn

    def init_empty_model(self, device='cpu'):
        self.ffn = self._materialize_module(self.ffn, device)

    def gen_input(self, bs: int = 8, seq_len: int = 512, device='cpu'):
        return self.__class__.to_device(torch.randn((bs, seq_len, self.hidden_size), dtype=self.dtype_), device)

    def profile(self, bs: int, seq_len: int, device='cpu', fwd_flag=True, profile_flag: str = 'time',
                skip_round: int = 10, test_round: int = 5, skip_init: bool = False, block_flag: bool = True):
        if not skip_init:
            self.init_empty_model(device='cpu')
        hidden_states = self.gen_input(bs, seq_len, 'cpu' if not skip_init else device)

        def input_factory(target_device, is_forward):
            return (self._detach_to_device(hidden_states, target_device, requires_grad=not is_forward),), {}

        scale_factor = 1 if block_flag else self.layer
        run_time, model_mem, act_mem, total_mem = self._profile_module(
            self.ffn,
            input_factory=input_factory,
            device=device,
            fwd_flag=fwd_flag,
            profile_flag=profile_flag,
            skip_round=skip_round,
            test_round=test_round,
            scale_factor=scale_factor,
        )

        if self.verbose:
            print(f"{self.model_id} | ffn | batch size {bs} | seq_len {seq_len} | layers {self.layer} | {self.dtype_} | {'forward' if fwd_flag else 'backward'}")
            if profile_flag == 'time':
                print(f"{'block' if block_flag else 'model'} ffn runing time: {run_time:.5f} s")
            else:
                if block_flag:
                    print(f'block ffn model memory: {model_mem:.4f} GB')
                    print(f'block ffn activation memory: {act_mem:.4f} GB')
                    print(f'block ffn total memory: {total_mem:.4f} GB')
                else:
                    print(f'model ffn memory: {model_mem / self.layer:.4f}/{model_mem:.4f} GB')
                    print(f'model ffn activation memory: {act_mem / self.layer:.4f}/{act_mem:.4f} GB')
                    print(f'model ffn total memory: {total_mem / self.layer:.4f}/{total_mem:.4f} GB')
        return run_time, model_mem, act_mem, total_mem

    def peftModel(self, config: PeftConfig = None):
        raise NotImplementedError('FFNProfiler does not support PEFT profiling.')

def test():
    '''
        Test the ProfileModel class's method.
    '''
    bs, seq, device, rank, lora_alpha, lora_dropout = 8, 512, 'npu', 8, 16, 0.05
    path = '/home/zjh/code/HF-LLM-Profiler/model/Qwen2.5-3B' # Qwen2.5-3B Falcon3-10B-Base
    profiler = ModelProfiler(path, dtype=torch.float32, verbose=True, device=device)
    profiler.peftModel()
    # print(profiler.trans)
    print(f'------------------{profiler.model_id} Profile memory------------------')
    # Test forward memory
    getattr(torch, profiler.backend).empty_cache()
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, True, 'memory', 50, 100))
    thread.start()
    thread.join()
    # Test backward memory
    getattr(torch, profiler.backend).empty_cache()
    thread = multiprocessing.Process(target=profiler.profile, args=(bs, seq, device, False, 'memory', 50, 100))
    thread.start()
    thread.join()

if __name__ == '__main__':
    test()
