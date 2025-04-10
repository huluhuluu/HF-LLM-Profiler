from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
from torch import nn
from calflops import calculate_flops
import time

model_path = '/data/HF_MODELS'

# model_id, target_module = 'gpt2-xl', None
# seq_len, hidden_size, batch_size, layers = 512, 1600, 8, 48

# model_id, target_module = 'Qwen2.5-3B', None
# seq_len, hidden_size, batch_size, layers = 512, 2048, 8, 36

# model_id, target_module = 'Llama-3.1-8B', None
# seq_len, hidden_size, batch_size, layers = 512, 4096, 8, 32

# model_id, target_module = 'gemma-3-12b-it', None
# seq_len, hidden_size, batch_size, layers = 512, 4096, 8, 32

# model_id, target_module = 'Mistral-7B-v0.1', None
# seq_len, hidden_size, batch_size, layers= 512, 4096, 8, 32

# model_id, target_module = 'Falcon3-10B-Base', None
# seq_len, hidden_size, batch_size, layers = 512, 3072, 8, 40

# model_id, target_module = 'starcoder2-15b', None
# seq_len, hidden_size, batch_size, layers = 512, 6144, 8, 40

# model_id, target_module = 'QwQ-32B', None
# seq_len, hidden_size, batch_size, layers = 512, 5120, 8, 64

# model_id, target_module = 'gemma-2-27b', None
# seq_len, hidden_size, batch_size, layers = 512, 4608, 8, 46

# model_id, target_module = 'Yi-1.5-34B', None
# seq_len, hidden_size, batch_size, layers = 512, 7168, 8, 60

# model_id, target_module = 'deepseek-coder-33b-instruct', None
# seq_len, hidden_size, batch_size, layers = 512, 7168, 8, 62

model_id, target_module = 'phi-4', None
seq_len, hidden_size, batch_size, layers = 512, 5120, 8, 40

# model_id, target_module = 'genz-70b', None
# seq_len, hidden_size, batch_size, layers = 512, 7168, 8, 62

# profile time to set 
lora_rank, lora_alpha = 32, 64
num_labels = 20
gpu_id = 1
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
load_pth_tag, save_pth_tag = True, False

def main():
    tokenizer = AutoTokenizer.from_pretrained(f'{model_path}/{model_id}')
    tokenizer.pad_token = tokenizer.eos_token

    if load_pth_tag:
        model = torch.load(f'model/{model_id}_blocks.pth', weights_only=False)
        embeds = torch.load(f'model/{model_id}_embeds.pth', weights_only=False)
        rotate = torch.load(f'model/{model_id}_rotate.pth', weights_only=False)
    else:
        # get pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(
            f'{model_path}/{model_id}', num_labels=10
        )
        # print(model)
        
        rank, lora_alpha = 8, 16
        if 'starcoder2-15b' == model_id:
            lora_config = LoraConfig(
                r = rank,
                lora_alpha = lora_alpha,
                target_modules = ['q_proj', 'v_proj'],
                lora_dropout = 0.05,
                bias = "none"
            )
        elif 'phi-4' == model_id:
            lora_config = LoraConfig(
                r = rank,
                lora_alpha = lora_alpha,
                target_modules = ['o_proj'],
                lora_dropout = 0.05,
                bias = "none"
            )
        else:
            # Define LoRA Config
            lora_config = LoraConfig(
                r = rank,
                lora_alpha = lora_alpha,
                lora_dropout = 0.05,
                bias = "none"
            )
        # apply lora to bert model
        model = get_peft_model(model, lora_config)
        # print(model)
        # exit()
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f'{model_id} | {layers} layers')
    print(f'batch size {batch_size} | seq_len {seq_len}')
    print(device, torch.cuda.get_device_name())

    if not load_pth_tag:
        embeds = model.base_model.model.model.embed_tokens
        rotate = model.base_model.model.model.rotary_emb
        # get stages
        if 'bert-large' in model_id:
            model = model.base_model.model.bert.encoder.layer[0]
        elif 'roberta-large' in model_id:
            model = model.base_model.model.roberta.encoder.layer[0]
        elif 't5-3b' in model_id:
            # model = model.base_model.model.transformer.encoder.block[1]
            model = model.base_model.model.transformer.decoder.block[1]
        elif 'gpt2-xl' in model_id:
            model = model.base_model.model.transformer.h[0]
        elif model_id in ['Qwen2.5-3B', 'Llama-3.1-8B', 'gemma-3-12b-it', 'Mistral-7B-v0.1', 'Falcon3-10B-Base', 
                          'genz-70b', 'phi-4',
                          'QwQ-32B', 'gemma-2-27b', 'Yi-1.5-34B','deepseek-coder-33b-instruct']:
            # latest model using same structure
            model = model.base_model.model.model.layers[1]

        # print(model)
        if save_pth_tag:
            torch.save(model, f'model/{model_id}_blocks.pth')
            torch.save(embeds, f'model/{model_id}_embeds.pth')
            torch.save(rotate, f'model/{model_id}_rotate.pth')
    # print('------------------------------')
    # model.print_trainable_parameters()
    # print('------------------------------')

    # get flops of model
    print(f'{model_id} | batch size {batch_size} | seq_len {seq_len} | layers {layers}')
    model, embeds, rotate = model.to(device), embeds.to(device), rotate.to(device)
    input_ids_shape = (batch_size, seq_len)
    input_shape, attention_shape  = [batch_size, seq_len, hidden_size], [batch_size, seq_len]

    input_ids = torch.ones(input_ids_shape, dtype=torch.int64).to(device)
    position_ids = torch.arange(0, input_shape[1], dtype=torch.int64).unsqueeze(0).to(device)
    attention_mask = torch.ones(attention_shape, dtype=torch.int64)
    def mask(mask_):
        if mask_.dim() == 3:
            mask_ = mask_[:, None, :, :]
        elif mask_.dim() == 2:
            mask_ = mask_[:, None, None, :]
        mask_ = (1.0 - mask_) * torch.finfo(torch.float).min
        return mask_
    attention_mask = mask(attention_mask).to(device)

    hidden_state = embeds(input_ids)
    input_embeds = rotate(hidden_state, position_ids)
    print(input_embeds[0].shape, input_embeds[0].dtype)
    print(input_embeds[1].shape, input_embeds[1].dtype)

    kwargs = {
        'position_embeddings': input_embeds,
        'attention_mask': attention_mask,
        'hidden_states': hidden_state
    }
    # modify /root/miniconda3/envs/profile/lib/python3.12/site-packages/calflops/flops_counter.py
    # line 152(due to rotary embedding)
    # if kwargs:
    #     for key, value in kwargs.items():
    #         if type(value) is tuple: # modfied
    #             continue
    #         else:
    #             kwargs[key] = value.to(device)
    flops, macs, params = calculate_flops(model = model,
                                          kwargs = kwargs,
                                        print_results = True
                                        )

if __name__ == '__main__':
    main()