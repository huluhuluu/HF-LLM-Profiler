from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel
import torch
from calflops import calculate_flops
MODEL_LAYER_KEY = {'default': 'num_hidden_layers'}
MODEL_HIDEEN_SIZE_KEY = {'default': 'hidden_size'}
MODEL_DTYPE_KEY = {'default': 'torch_dtype'}
MODEL_EMBED_KEY = {'default': 'embed_tokens'}
MODEL_ROTATE_KEY = {'default': 'rotary_emb'}
MODEL_TRANS_KEY = {'default': 'layers'}

class ProfileModel(object):
    def __init__(self, model_id_or_path: str):
        self.model_id_or_path = model_id_or_path
        self.config = AutoConfig.from_pretrained(model_id_or_path)
        self.model_id = self.get_model_id(model_id_or_path)
        
        self.model, self.trans, self.embeds, self.rotate = self.get_model(self.config) # get the model from the config
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

    def get_model(self, config):
        '''
            Get the empty model from the config.
            Get the transformer/embeds/rotate layer from the empty model.
        '''
        # empty model
        with init_empty_weights():
            model = AutoModel.from_config(config)

        global MODEL_EMBED_KEY, MODEL_ROTATE_KEY, MODEL_TRANS_KEY
        # get only a singel layer of the transformer
        return model, self.get_attr(MODEL_TRANS_KEY, model, self.model_id)[1], \
                      self.get_attr(MODEL_EMBED_KEY, model, self.model_id), \
                      self.get_attr(MODEL_ROTATE_KEY, model, self.model_id)

    @classmethod
    def count_param(cls, model):
        '''
            Count the number of parameters and trainable parameters in the model both metadata and actual data form.
        '''
        meta_total_params = sum(p.numel() for p in model.parameters() if p.device == torch.device('meta'))
        meta_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad and p.device == torch.device('meta'))
        actual_total_params = sum(p.numel() for p in model.parameters())
        actual_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f'Trainable params(actual|meta): {actual_trainable_params / 1e9:.2f}B|{meta_trainable_params / 1e9:.2f}B')
        print(f'Total params(actual|meta): {actual_total_params / 1e9:.2f}B|{meta_total_params / 1e9:.2f}B')
        return actual_trainable_params, meta_trainable_params, actual_total_params, meta_total_params
        
    def init_empty_model(self, device = 'cpu'):
        '''
            Initialize the empty model we got with metedata form.
        '''
        # self.model = self.model.to(device)
        self.trans = self.trans.to_empty(device=device)
        self.embeds = self.embeds.to_empty(device=device)
        self.rotate = self.rotate.to_empty(device=device)
    
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
        '''
        # gen input for the model transformer
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
        
        print('\n-------------------------------')
        print(f'{self.model_id} | batch size {bs} | seq_len {seq_len} | layers {self.layer}')
        
        fwd_flops, bwd_flops, param = scale_up(fwd_trans_flops, self.layer), \
                                      scale_up(bwd_trans_flops, self.layer), \
                                      scale_up(trans_params, self.layer)

        print(f'forward flops: {fwd_flops} | backward flops: {bwd_flops} | params: {param}')

    # TODO:
    def profile_forward(self, bs: int, seq_len: int, device = 'cpu'):
        '''
            Profile the forward pass of the model.
            This is a placeholder function and should be implemented in the future.
        '''
        pass
    
    # TODO:
    def profile_backward(self, bs: int, seq_len: int, device = 'cpu'):
        '''
            Profile the backward pass of the model.
            This is a placeholder function and should be implemented in the future.
        '''
        pass

if __name__ == '__main__':
    path = '/data/HUGGINGFACE/Qwen2.5-3B' # Qwen2.5-3B Falcon3-10B-Base
    profiler = ProfileModel(path)
    # print('------------------------------')
    # print(profiler.config)
    # print(profiler.config.num_hidden_layers)
    # print(profiler.config.hidden_size)
    # print(profiler.config.torch_dtype)
    # print(profiler.model_id)
    # print(profiler.layer)
    # print(profiler.hidden_size)
    # print(profiler.model, end='\n------------------\n')
    # print(profiler.trans, end='\n------------------\n')
    # print(profiler.embeds, end='\n------------------\n')
    # print(profiler.rotate, end='\n------------------\n')
    # print(profiler.count_param(profiler.model))
    # print('------------before init---------------')
    # for name, param in profiler.trans.named_parameters():
    #     print(f"Parameter: {name} | dtype: {param.dtype} | device: {param.device}")
    # print(profiler.count_param(profiler.trans), end='\n------------------\n')
    # print(profiler.count_param(profiler.embeds), end='\n------------------\n')
    # print(profiler.count_param(profiler.rotate), end='\n------------------\n')

    profiler.init_empty_model('cuda:0')
    
    # print('------------after init---------------')
    # for name, param in profiler.trans.named_parameters():
    #     print(f"Parameter: {name} | dtype: {param.dtype} | device: {param.device}")
    # print(profiler.count_param(profiler.trans), end='\n------------------\n')
    # print(profiler.count_param(profiler.embeds), end='\n------------------\n')
    # print(profiler.count_param(profiler.rotate), end='\n------------------\n')
    profiler.get_calflops(8, 512, 'cuda:0')
    # TODO: test getattr recursive .