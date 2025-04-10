from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
import time
import inspect

model_path = '/data/HUGGINGFACE'
gpu_id = 0
# model_id, target_module = 'bert-large-uncased', ["query", "value"]
# model_id, target_module = 'roberta-large', ["query", "value"]
# model_id, target_module = 't5-3b', ["q", "v"]
# seq_len, hidden_size, batch_size = 512, 1024, 8

model_id, target_module = 'gpt2-xl', None
seq_len, hidden_size, batch_size = 512, 1600, 2

# model_id, target_module = 'Qwen2.5-3B', None
# seq_len, hidden_size, batch_size = 1024, 2048, 2

# model_id, target_module = 'Llama-3.1-8B', None
# seq_len, hidden_size, batch_size = 128, 4096, 8

# using saved pth file(some device can not load the full model)
load_pth_tag, save_pth_tag = True, False

class GPTEmbedding(torch.nn.Module):
    def __init__(self, wte, wpe, drop):
        super(GPTEmbedding, self).__init__()
        self.wte = wte
        self.wpe = wpe
        self.drop = drop

    def forward(self, input_ids, past_length = 0):
        device = input_ids.device
        position_ids = torch.arange(past_length, input_ids.size()[-1] + past_length, dtype=torch.long, device=device)
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        return self.drop(hidden_states)
    
def profile_forward(model, skip_round, test_round, device, input_shape):
    # set device
    model.to(device)

    # train and compute avg time
    # to compute the average time for training, must replace modeling_bert.py to your
    begin_time, end_time = 0, 0

    input_ =  torch.ones(input_shape, dtype=torch.int32).to(device)
    with torch.no_grad():
        for epoch in range(test_round + skip_round):
            if epoch == skip_round:
                begin_time = time.time()

            # forward
            tensor = model(input_)
    end_time = time.time()
    # avg time
    print(f"forward avg time: {(end_time - begin_time) / test_round} s")
    print(f"forward use memeory {torch.cuda.max_memory_allocated(device) / (1024 ** 3)} GB")

def main():
    global model_path, gpu_id, seq_len, hidden_size, batch_size
    global model_id, target_module

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    input_shape = [batch_size, seq_len]
    
    if load_pth_tag:
        model = torch.load(f'model/{model_id}_embedding.pth')
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
            model = model.base_model.model.bert.embeddings
        elif 'roberta-large' in model_id:
            model = model.base_model.model.roberta.embeddings
        elif 't5-3b' in model_id:
            model = model.base_model.model.transformer.embeddings
        elif 'gpt2-xl' in model_id:
            model = GPTEmbedding(
                model.base_model.model.transformer.wte,
                model.base_model.model.transformer.wpe,
                model.base_model.model.transformer.drop
            )
        elif 'Qwen2.5-3B' in model_id:
            model = model.base_model.model.model.embed_tokens
        elif 'Llama-3.1-8B' in model_id:
            model = model.base_model.model.model.embed_tokens
        # print(model)
        if save_pth_tag:
            torch.save(model, f'model/{model_id}_embedding.pth')
    # print(model)
    
    skip_round, test_round = 100, 1000
    # forward test
    torch.cuda.empty_cache()
    profile_forward(model, skip_round, test_round, device, input_shape)

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