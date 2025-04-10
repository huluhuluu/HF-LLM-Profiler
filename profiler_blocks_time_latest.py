from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
import time
import copy

model_path = '/data/HF_MODELS'
gpu_id = 2
block_num = 1
# model_id, target_module = 'gpt2-xl', None
# seq_len, hidden_size, batch_size, layers = 512, 1600, 8, 48

# model_id, target_module = 'Qwen2.5-3B', None
# seq_len, hidden_size, batch_size, layers = 512, 2048, 8, 36

# model_id, target_module = 'Llama-3.1-8B', None
# seq_len, hidden_size, batch_size, layers = 512, 4096, 8, 32

model_id, target_module = 'Mistral-7B-v0.1', None
seq_len, hidden_size, batch_size, layers= 512, 4096, 8, 32

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

# model_id, target_module = 'phi-4', None
# seq_len, hidden_size, batch_size, layers = 512, 5120, 8, 40

# using saved pth file(some device can not load the full model)
load_pth_tag, save_pth_tag = True, False

def profile_forward(model, skip_round, test_round, device, input_ids, attention_mask, input_embeds, block_num = 5):
    models = [copy.deepcopy(model).to(device) for _ in range(block_num)]

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    begin_time, end_time = 0, 0

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    input_embeds = (input_embeds[0].to(device), input_embeds[1].to(device))
    def mask(mask_):
        if mask_.dim() == 3:
            mask_ = mask_[:, None, :, :]
        elif mask_.dim() == 2:
            mask_ = mask_[:, None, None, :]
        mask_ = (1.0 - mask_) * torch.finfo(torch.float).min
        return mask_
    attention_mask = mask(attention_mask)

    with torch.no_grad():
        for epoch in range(test_round + skip_round):
            if epoch == skip_round:
                begin_time = time.time()
            
            tensor = input_ids
            # forward
            for model in models:
                if model_id == 'gpt2-xl':
                    for _ in range(block_num):
                        tensor = model(tensor, attention_mask = attention_mask)[0]
                else:
                    for _ in range(block_num):
                        tensor = model(tensor, attention_mask = attention_mask, position_embeddings = input_embeds)[0]

    end_time = time.time()
    # avg time
    print(f"forward avg time: {(end_time - begin_time) / test_round / block_num} s")
    print(f"forward use memeory {torch.cuda.max_memory_allocated(device) / (1024 ** 3)} GB")

def profile_backward(model, skip_round, test_round, device, input_ids, attention_mask, input_embeds, block_num = 5):
    # set device
    models = [copy.deepcopy(model).to(device) for _ in range(block_num)]

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    
    begin_time, end_time = 0, 0

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    input_embeds = (input_embeds[0].to(device), input_embeds[1].to(device))
    def mask(mask_):
        if mask_.dim() == 3:
            mask_ = mask_[:, None, :, :]
        elif mask_.dim() == 2:
            mask_ = mask_[:, None, None, :]
        mask_ = (1.0 - mask_) * torch.finfo(torch.float).min
        return mask_
    attention_mask = mask(attention_mask)

    for epoch in range(test_round + skip_round * 2):
        if epoch == skip_round:
            begin_time = time.time()

        tensor = input_ids
        model.zero_grad()
        # forward
        for model in models:
            if model_id == 'gpt2-xl':
                for _ in range(block_num):
                    tensor = model(tensor, attention_mask = attention_mask)[0]
            else:
                for _ in range(block_num):
                    tensor = model(tensor, attention_mask = attention_mask, position_embeddings = input_embeds)[0]
        # backward
        torch.autograd.backward(tensor[0] if type(tensor) is tuple else tensor, grad_tensors=torch.ones_like(tensor[0] if type(tensor) is tuple else tensor))
        if epoch == skip_round + test_round:
            end_time = time.time()
    # avg time
    print(f"backward avg time: {(end_time - begin_time) / test_round / block_num} s")
    print(f"backward use memeory {torch.cuda.max_memory_allocated(device) / (1024 ** 3)} GB")

def profile_only_backward(model, skip_round, test_round, device, input_ids, attention_mask, input_embeds, block_num = 5):
    # set device
    models = [copy.deepcopy(model).to(device) for _ in range(block_num)]

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    
    begin_time, end_time = 0, 0

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    input_embeds = (input_embeds[0].to(device), input_embeds[1].to(device))
    def mask(mask_):
        if mask_.dim() == 3:
            mask_ = mask_[:, None, :, :]
        elif mask_.dim() == 2:
            mask_ = mask_[:, None, None, :]
        mask_ = (1.0 - mask_) * torch.finfo(torch.float).min
        return mask_
    attention_mask = mask(attention_mask)

    tensor = input_ids
    # forward
    for model in models:
        if model_id == 'gpt2-xl':
            for _ in range(block_num):
                tensor = model(tensor, attention_mask = attention_mask)[0]
        else:
            for _ in range(block_num):
                tensor = model(tensor, attention_mask = attention_mask, position_embeddings = input_embeds)[0]
    
    for epoch in range(test_round + skip_round):
        if epoch == skip_round:
            begin_time = time.time()

        model.zero_grad()
        
        # backward
        torch.autograd.backward(tensor[0] if type(tensor) is tuple else tensor, grad_tensors=torch.ones_like(tensor[0] if type(tensor) is tuple else tensor), retain_graph=True)

    end_time = time.time()
    # avg time
    print(f"backward only avg time: {(end_time - begin_time) / test_round / block_num} s")
    print(f"backward only use memeory {torch.cuda.max_memory_allocated(device) / (1024 ** 3)} GB")

def main():
    global model_path, gpu_id, seq_len, hidden_size, batch_size
    global model_id, target_module, block_num, layers

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    input_shape, attention_shape  = [batch_size, seq_len, hidden_size], [batch_size, seq_len]
    
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

    print(f'{model_id} | {layers} layers')
    print(f'batch size {batch_size} | seq_len {seq_len}')
    print(device, torch.cuda.get_device_name())
    # get shape
    # input_ids =  torch.ones(input_shape, dtype=torch.int64)
    # print(type(model.base_model.model.model))
    # inputs_embeds = model.base_model.model.model.embed_tokens(input_ids)
    # position_ids = torch.arange(0, input_shape[1], dtype=torch.int64).unsqueeze(0)
    # t1, t2 = model.base_model.model.model.rotary_emb(inputs_embeds, position_ids)
    # print(t1.shape, t1.dtype)
    # print(t2.shape, t2.dtype)
    # <class 'transformers.models.llama.modeling_llama.LlamaModel'>
    # torch.Size([1, 512, 256]) torch.float32
    # torch.Size([1, 512, 256]) torch.float32
    # <class 'transformers.models.mistral.modeling_mistral.MistralModel'>
    # torch.Size([1, 512, 128]) torch.float32
    # torch.Size([1, 512, 128]) torch.float32
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
                          'genz-70b', 'phi-4'
                          'QwQ-32B', 'gemma-2-27b', 'Yi-1.5-34B','deepseek-coder-33b-instruct']:
            # latest model using same structure
            model = model.base_model.model.model.layers[1]

        # print(model)
        if save_pth_tag:
            torch.save(model, f'model/{model_id}_blocks.pth')
            torch.save(embeds, f'model/{model_id}_embeds.pth')
            torch.save(rotate, f'model/{model_id}_rotate.pth')
    
    # print(model)
    skip_round, test_round = 100, 200
    input_ids_shape = (batch_size, seq_len)
    input_ids = torch.ones(input_ids_shape, dtype=torch.int64)
    
    attention_mask = torch.ones(attention_shape, dtype=torch.int64)
    hidden_state = embeds(input_ids)
    position_ids = torch.arange(0, input_shape[1], dtype=torch.int64).unsqueeze(0)
    input_embeds = rotate(hidden_state, position_ids)
    print(input_embeds[0].shape, input_embeds[0].dtype)
    print(input_embeds[1].shape, input_embeds[1].dtype)

    # forward test
    torch.cuda.empty_cache()
    # profile_forward(model, skip_round, test_round, device, hidden_state, attention_mask, input_embeds, block_num)

    # backward test
    torch.cuda.empty_cache()
    profile_backward(model, skip_round, test_round, device, hidden_state.requires_grad_(True), attention_mask, input_embeds, block_num)

    # only backward test
    torch.cuda.empty_cache()
    # profile_only_backward(model, skip_round, test_round, device, hidden_state.requires_grad_(True), attention_mask, input_embeds, block_num)

# use multiprocessing to test all possible hyper-parameters
def auto_test():
    import multiprocessing
    model_list = ['gpt2-xl', 'Qwen2.5-3B', 'Llama-3.1-8B']
    hidden_dict = {'gpt2-xl':1600, 'Qwen2.5-3B':2048, 'Llama-3.1-8B':4096}
    seq_len_list = [64, 128, 256, 512]
    batch_size_list = [2, 8]
    block_num_dict = {
        'gpt2-xl':{
            2: {64:12, 128:8, 256:6, 512:4 },
            8: {64:6, 128:4, 256:3, 512:2 }
        },
        'Qwen2.5-3B':{
            2: {64:8, 128:6, 256:5, 512:3 },
            8: {64:5, 128:3, 256:2, 512:1 }
        },
        'Llama-3.1-8B':{
            2: {64:4, 128:3, 256:3, 512:2 },
            8: {64:3, 128:2, 256:1, 512:1 }
        }
    }

    # travese all possible hype-param
    for m in model_list:
        for s in seq_len_list:
            for b in batch_size_list:
                global seq_len, hidden_size, batch_size, model_id, block_num
                model_id, seq_len, batch_size = m, s, b
                hidden_size = hidden_dict[m]
                block_num = block_num_dict[model_id][batch_size][seq_len]
                while True:
                    torch.cuda.empty_cache()
                    thread = multiprocessing.Process(target=main)
                    thread.start()
                    thread.join()
                    if thread.exitcode == 0:
                        break
                    else:
                        block_num -= 1


def run():
    import multiprocessing
    model_list = [ 'Falcon3-10B-Base', 'phi4', 'starcoder2-15b'] # 'Mistral-7B-v0.1',
    hidden_dict = {'Mistral-7B-v0.1':4096, 'Falcon3-10B-Base':3072, 'phi4':5120, 'starcoder2-15b':6144}
    
    global seq_len, hidden_size, batch_size, model_id, block_num
    seq_len, batch_size, block_num = 512, 8, 1

    # travese all possible hype-param
    for m in model_list:
        model_id = m
        hidden_size = hidden_dict[m]
        while True:
            torch.cuda.empty_cache()
            thread = multiprocessing.Process(target=main)
            thread.start()
            thread.join()
            if thread.exitcode == 0:
                break
            else:
                exit(1)

if __name__ == '__main__':
    # auto_test()
    main()
    # run()

# phi-4 | 40 layers
# batch size 8 | seq_len 512
# cuda:0 NVIDIA GeForce RTX 4090
# torch.Size([1, 512, 128]) torch.float32
# torch.Size([1, 512, 128]) torch.float32
# backward avg time: 0.21783329248428346 s
# backward use memeory 3.6502914428710938 GB

# starcoder2-15b | 40 layers
# batch size 8 | seq_len 512
# cuda:0 NVIDIA GeForce RTX 4090
# torch.Size([1, 512, 128]) torch.float32
# torch.Size([1, 512, 128]) torch.float32
# backward avg time: 0.2347813391685486 s
# backward use memeory 3.557292938232422 GB

# Falcon3-10B-Base | 40 layers
# batch size 8 | seq_len 512
# cuda:0 NVIDIA GeForce RTX 4090
# torch.Size([1, 512, 256]) torch.float32
# torch.Size([1, 512, 256]) torch.float32
# backward avg time: 0.14431448340415953 s
# backward use memeory 3.5039749145507812 GB

# Mistral-7B-v0.1 | 32 layers
# batch size 8 | seq_len 512
# cuda:0 NVIDIA GeForce RTX 4090
# torch.Size([1, 512, 128]) torch.float32
# torch.Size([1, 512, 128]) torch.float32
# backward avg time: 0.15679063200950621 s
# backward use memeory 2.7988128662109375 GB