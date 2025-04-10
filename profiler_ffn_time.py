from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
import time

model_path = '/data/HUGGINGFACE'
gpu_id = 0
# model_id, target_module = 'bert-large-uncased', ["query", "value"]
# model_id, target_module = 'roberta-large', ["query", "value"]
# model_id, target_module = 't5-3b', ["q", "v"]
# seq_len, hidden_size, num_label, batch_size = 512, 1024, 20, 2

# model_id, target_module = 'gpt2-xl', None
# seq_len, hidden_size, num_label, batch_size = 128, 1600, 20, 2

# model_id, target_module = 'Qwen2.5-3B', None
# seq_len, hidden_size, num_label, batch_size = 128, 2048, 20, 2

model_id, target_module = 'Llama-3.1-8B', None
seq_len, hidden_size, num_label, batch_size = 128, 4096, 20, 2
# using saved pth file(some device can not load the full model)
load_pth_tag, save_pth_tag = True, False

def profile_forward(models, skip_round, test_round, device, input_shape, label_shape):
    # set device
    for model in models:
        model.to(device)

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    begin_time, end_time = 0, 0

    input_ = torch.ones(input_shape, dtype=torch.float32).to(device)
    label = torch.ones(label_shape, dtype=torch.int64).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for epoch in range(test_round + skip_round):
            if epoch == skip_round:
                begin_time = time.time()

            tensor = input_
            # forward
            tensor = models(tensor[0] if type(tensor) is tuple else tensor)

            if type(tensor) is tuple:
                tensor = tensor[0]
            if len(tensor.size()) == 3:
                tensor = tensor[:, -1]
            tensor = loss_func(tensor, label)

    end_time = time.time()
    # avg time
    print(f"forward avg time: {(end_time - begin_time) / test_round} s")
    print(f"forward use memeory {torch.cuda.max_memory_allocated(device) / (1024 ** 3)} GB")

def profile_backward(models, skip_round, test_round, device, input_shape, label_shape):
    # set device
    for model in models:
        model.to(device)

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    
    begin_time, end_time = 0, 0

    input_ =  torch.ones(input_shape, dtype=torch.float32).requires_grad_(True).to(device)
    label = torch.ones(label_shape, dtype=torch.int64).to(device)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(test_round + skip_round):
        if epoch == skip_round:
            begin_time = time.time()

        model.zero_grad()
        # forward
        tensor = input_
        # forward
        tensor = models(tensor[0] if type(tensor) is tuple else tensor)
        
        if type(tensor) is tuple:
            tensor = tensor[0]
        if len(tensor.size()) == 3:
            tensor = tensor[:, -1]
        tensor = loss_func(tensor, label)
        
        # backward
        torch.autograd.backward(tensor[0] if type(tensor) is tuple else tensor)

    end_time = time.time()
    # avg time
    print(f"backward avg time: {(end_time - begin_time) / test_round} s")
    print(f"backward use memeory {torch.cuda.max_memory_allocated(device) / (1024 ** 3)} GB")

def profile_only_backward(models, skip_round, test_round, device, input_shape, label_shape):
    # set device
    for model in models:
        model.to(device)

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    
    begin_time, end_time = 0, 0

    input_ =  torch.ones(input_shape, dtype=torch.float32).requires_grad_(True).to(device)
    label = torch.ones(label_shape, dtype=torch.int64).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    # forward
    tensor = input_
    tensor = models(tensor[0] if type(tensor) is tuple else tensor)
    if type(tensor) is tuple:
        tensor = tensor[0]
    if len(tensor.size()) == 3:
        tensor = tensor[:, -1]
    tensor = loss_func(tensor, label)

    for epoch in range(test_round + skip_round):
        if epoch == skip_round:
            begin_time = time.time()

        model.zero_grad()
        
        # backward
        torch.autograd.backward(tensor[0] if type(tensor) is tuple else tensor, retain_graph=True)

    end_time = time.time()
    # avg time
    print(f"backward only avg time: {(end_time - begin_time) / test_round} s")
    print(f"backward only use memeory {torch.cuda.max_memory_allocated(device) / (1024 ** 3)} GB")

def main():
    global model_path, gpu_id, seq_len, hidden_size, batch_size
    global model_id, target_module, num_label

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    input_shape, label_shape  = [batch_size, seq_len, hidden_size], [batch_size]
    
    if load_pth_tag:
        models = torch.load(f'model/{model_id}_ffn.pth')
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
        models = torch.nn.Sequential()
        if 'bert-large' in model_id:
            models.append(model.base_model.model.bert.pooler)
            models.append(model.base_model.model.dropout)
            models.append(model.base_model.model.classifier)
        elif 'roberta-large' in model_id:
            models.append(model.base_model.model.classifier)
        elif 't5-3b' in model_id:
            pass
        elif 'gpt2-xl' in model_id:
            models.append(model.base_model.model.transformer.ln_f)
            models.append(model.base_model.model.score)
        elif 'Qwen2.5-3B' in model_id:
            models.append(model.base_model.model.model.norm)
            models.append(model.base_model.model.score)
        elif 'Llama-3.1-8B' in model_id:
            models.append(model.base_model.model.model.norm)
            models.append(model.base_model.model.score)
        # print(model)
        if save_pth_tag:
            torch.save(models, f'model/{model_id}_ffn.pth')
    # print(models)
    
    skip_round, test_round = 100, 1000
    # forward test
    torch.cuda.empty_cache()
    profile_forward(models, skip_round, test_round, device, input_shape, label_shape)

    # backward test
    torch.cuda.empty_cache()
    profile_backward(models, skip_round, test_round, device, input_shape, label_shape)

    # only backward test
    torch.cuda.empty_cache()
    profile_only_backward(models, skip_round, test_round, device, input_shape, label_shape)

# use multiprocessing to test all possible hyper-parameters
def auto_test():
    import multiprocessing
    model_list = ['gpt2-xl', 'Qwen2.5-3B', 'Llama-3.1-8B']
    hidden_dict = {'gpt2-xl':1600, 'Qwen2.5-3B':2048, 'Llama-3.1-8B':4096}
    seq_len_list = [64, 128, 256, 512]
    batch_size_list = [2, 8]
    # travese all possible hype-param
    for m in model_list:
        for s in seq_len_list:
            for b in batch_size_list:
                global seq_len, hidden_size, batch_size, model_id
                model_id, seq_len, batch_size = m, s, b
                hidden_size = hidden_dict[m]
                torch.cuda.empty_cache()
                thread = multiprocessing.Process(target=main)
                thread.start()
                thread.join()

if __name__ == '__main__':
    auto_test()
    # main()