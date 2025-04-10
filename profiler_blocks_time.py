from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
import time
import copy

model_path = '/data/HUGGINGFACE'
gpu_id = 0
# model_id, target_module = 'bert-large-uncased', ["query", "value"]
# model_id, target_module = 'roberta-large', ["query", "value"]
# model_id, target_module = 't5-3b', ["q", "v"]
# seq_len, hidden_size, batch_size = 512, 1024, 8

block_num = 6
model_id, target_module = 'gpt2-xl', None
seq_len, hidden_size, batch_size = 64, 1600, 8

# model_id, target_module = 'Qwen2.5-3B', None
# seq_len, hidden_size, batch_size = 256, 2048, 8

# model_id, target_module = 'Llama-3.1-8B', None
# seq_len, hidden_size, batch_size = 256, 4096, 8

# using saved pth file(some device can not load the full model)
load_pth_tag, save_pth_tag = True, False

def profile_forward(model, skip_round, test_round, device, input_shape, attention_shape, block_num = 5):
    models = [copy.deepcopy(model).to(device) for _ in range(block_num)]

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    begin_time, end_time = 0, 0

    input_ =  torch.ones(input_shape, dtype=torch.float32).to(device)
    attention_mask = torch.ones(attention_shape, dtype=torch.int64).to(device)
    position_ids = torch.arange(0, input_shape[1], dtype=torch.int64).unsqueeze(0).to(device)
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
            
            tensor = input_
            # forward
            for model in models:
                if model_id == 'gpt2-xl':
                    for _ in range(block_num):
                        tensor = model(tensor, attention_mask = attention_mask)[0]
                else:
                    for _ in range(block_num):
                        tensor = model(tensor, attention_mask = attention_mask, position_ids = position_ids)[0]

    end_time = time.time()
    # avg time
    print(f"forward avg time: {(end_time - begin_time) / test_round / block_num} s")
    print(f"forward use memeory {torch.cuda.max_memory_allocated(device) / (1024 ** 3)} GB")

def profile_backward(model, skip_round, test_round, device, input_shape, attention_shape, block_num = 5):
    # set device
    models = [copy.deepcopy(model).to(device) for _ in range(block_num)]

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    
    begin_time, end_time = 0, 0

    input_ =  torch.ones(input_shape, dtype=torch.float32).requires_grad_(True).to(device)
    attention_mask = torch.ones(attention_shape, dtype=torch.int64).to(device)
    position_ids = torch.arange(0, input_shape[1], dtype=torch.int64).unsqueeze(0).to(device)
    def mask(mask_):
        if mask_.dim() == 3:
            mask_ = mask_[:, None, :, :]
        elif mask_.dim() == 2:
            mask_ = mask_[:, None, None, :]
        mask_ = (1.0 - mask_) * torch.finfo(torch.float).min
        return mask_
    attention_mask = mask(attention_mask)

    for epoch in range(test_round + skip_round):
        if epoch == skip_round:
            begin_time = time.time()

        tensor = input_
        model.zero_grad()
        # forward
        for model in models:
            if model_id == 'gpt2-xl':
                for _ in range(block_num):
                    tensor = model(tensor, attention_mask = attention_mask)[0]
            else:
                for _ in range(block_num):
                    tensor = model(tensor, attention_mask = attention_mask, position_ids = position_ids)[0]
        # backward
        torch.autograd.backward(tensor[0] if type(tensor) is tuple else tensor, grad_tensors=torch.ones_like(tensor[0] if type(tensor) is tuple else tensor))
    end_time = time.time()
    # avg time
    print(f"backward avg time: {(end_time - begin_time) / test_round / block_num} s")
    print(f"backward use memeory {torch.cuda.max_memory_allocated(device) / (1024 ** 3)} GB")

def profile_only_backward(model, skip_round, test_round, device, input_shape, attention_shape, block_num = 5):
    # set device
    models = [copy.deepcopy(model).to(device) for _ in range(block_num)]

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    
    begin_time, end_time = 0, 0

    input_ =  torch.ones(input_shape, dtype=torch.float32).requires_grad_(True).to(device)
    attention_mask = torch.ones(attention_shape, dtype=torch.int64).to(device)
    position_ids = torch.arange(0, input_shape[1], dtype=torch.int64).unsqueeze(0).to(device)
    def mask(mask_):
        if mask_.dim() == 3:
            mask_ = mask_[:, None, :, :]
        elif mask_.dim() == 2:
            mask_ = mask_[:, None, None, :]
        mask_ = (1.0 - mask_) * torch.finfo(torch.float).min
        return mask_
    attention_mask = mask(attention_mask)

    tensor = input_
    # forward
    for model in models:
        if model_id == 'gpt2-xl':
            for _ in range(block_num):
                tensor = model(tensor, attention_mask = attention_mask)[0]
        else:
            for _ in range(block_num):
                tensor = model(tensor, attention_mask = attention_mask, position_ids = position_ids)[0]
    
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
    global model_id, target_module, block_num

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    input_shape, attention_shape  = [batch_size, seq_len, hidden_size], [batch_size, seq_len]
    
    if load_pth_tag:
        model = torch.load(f'model/{model_id}_blocks.pth')
    else:
        # get pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(
            f'{model_path}/{model_id}', num_labels=10
        )
        # print(model)
        
        rank, lora_alpha = 8, 16
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

    print(model_id)
    print(f'batch size {batch_size} | seq_len {seq_len}')
    print(device, torch.cuda.get_device_name())

    if not load_pth_tag:
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
        elif 'Qwen2.5-3B' in model_id:
            model = model.base_model.model.model.layers[0]
        elif 'Llama-3.1-8B' in model_id:
            model = model.base_model.model.model.layers[0]
        # print(model)
        if save_pth_tag:
            torch.save(model, f'model/{model_id}_blocks.pth')
    
    # print(model)
    skip_round, test_round = 30, 70
    # forward test
    torch.cuda.empty_cache()
    profile_forward(model, skip_round, test_round, device, input_shape, attention_shape, block_num)

    # backward test
    torch.cuda.empty_cache()
    profile_backward(model, skip_round, test_round, device, input_shape, attention_shape, block_num)

    # only backward test
    torch.cuda.empty_cache()
    profile_only_backward(model, skip_round, test_round, device, input_shape, attention_shape, block_num)

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

if __name__ == '__main__':
    # auto_test()
    main()