from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
from torch import nn
from calflops import calculate_flops
import time

data_path = '/data/HF_MODELS/datasets'
cache_path = '/data/HF_MODELS'
model_id = 'genz-70b' 
# gpt2-xl Qwen2.5-3B  Llama-3.1-8B gemma-2-27b Mistral-7B-v0.1 starcoder2-15b QwQ-32B  
# deepseek-coder-33b-instruct Yi-1.5-34B  genz-70b phi-4
# genz-70b : pip install tiktoken blobfile
task = 'banking77'

# profile: local_bs to set 
batch_size, lr, epochs = 8, 0.01, 200
lora_rank, lora_alpha = 32, 64
num_labels = 20
max_seq_length = 512
task_to_keys = {
    "banking77": ("text", None),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "semeval": ("sentence", None),
}
# data = load_dataset('/data/HUGGINGFACE/data/sem_eval_2010_task_8')
# print(data)

gpu_id = 1
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(f'{cache_path}/{model_id}')

class DatasetSplit(Dataset):
    def __init__(self, dataset, task, idxs):
        self.dataset = dataset
        self.task = task
        self.idxs = [int(idx) for idx in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        global tokenizer, task_to_keys
        if task_to_keys[self.task][1] is not None:
            text_encoded = tokenizer(self.dataset[self.idxs[item]][task_to_keys[self.task][0]], self.dataset[self.idxs[item]][task_to_keys[self.task][1]],  truncation=True, padding='max_length', return_tensors='pt')
        else:
            text_encoded = tokenizer(self.dataset[self.idxs[item]][task_to_keys[self.task][0]], truncation=True, padding='max_length', return_tensors='pt')
        
        input_ids = text_encoded['input_ids'].squeeze()
        attention_mask = text_encoded['attention_mask'].squeeze()
        # print(input_ids, attention_mask, self.dataset[self.idxs[item]]['label'])
        return input_ids, attention_mask, self.dataset[self.idxs[item]]['label']


#data = load_dataset(f"data_path/{task}")

# make dataloader
# data_train = data['train']
# num_labels = len(data_train.features["label"].names)
# dataloader = DataLoader(DatasetSplit(data_train, task , [i for i in range(len(data_train))]), batch_size=batch_size, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(f'{cache_path}/{model_id}', num_labels=num_labels)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
# print(type(model))
# print(model)
# print('------------------------------')

# apply lora to bert model
print(model)
if 'starcoder2-15b' == model_id:
    lora_config = LoraConfig(
        r = lora_rank,
        lora_alpha = lora_alpha,
        target_modules = ['q_proj', 'v_proj'],
        lora_dropout = 0.05,
        bias = "none"
    )
elif 'phi-4' == model_id:
    lora_config = LoraConfig(
        r = lora_rank,
        lora_alpha = lora_alpha,
        target_modules = ['o_proj'],
        lora_dropout = 0.05,
        bias = "none"
    )
else:
    lora_config = LoraConfig(
        r = lora_rank,
        lora_alpha = lora_alpha,
        lora_dropout = 0.05,
        bias = "none"
    )
model = get_peft_model(model, lora_config)

# print(type(model))
print(model)
exit()
# print('------------------------------')
# model.print_trainable_parameters()
# print('------------------------------')

# get flops of model
print(f'{model_id} | batch size {batch_size} | seq_len {max_seq_length}')
flops, macs, params = calculate_flops(model = model, 
                                      input_shape = (batch_size, max_seq_length),
                                      transformer_tokenizer = tokenizer,
                                      print_results = True
                                    #   print_detailed = True
                                      )
exit()

# compute training time
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# print(model_id)
# print(f'batch size {batch_size}')
print(device, torch.cuda.get_device_name())
model.to(device)

# train and compute avg time
# to compute the average time for training, must replace modeling_bert.py to your
skip_round, test_round = 100, 1000
begin_time, end_time = 0, 0
iter = 0
for epoch in range(epochs):
    for batch, (input_ids, attention_mask, label) in enumerate(dataloader):
        if iter == skip_round:
            begin_time = time.time()
        elif iter == skip_round + test_round:
            end_time = time.time()
            print(f"avg time {(end_time - begin_time) / test_round} ")
            exit()
        input_ids, attention_mask, label = input_ids.to(device), attention_mask.to(device), label.to(device)

        model.zero_grad()
        # forward
        logits = model(input_ids, attention_mask=attention_mask).logits 
        loss = loss_func(logits, label)
        # backward
        # loss.backward()
        # optimizer.step()
        
        iter += 1
        # print("epoch: {}, batch: {}, loss: {}".format(epoch, batch, loss.item()))
        # if count > 100:
        #     print("gpu memory:")
        #     print(torch.cuda.max_memory_allocated(gpu_id) / 1024**2)
        #     exit()
        # 9060.75634765625 MB
        # backward avg time 0.17389991283416747
        # forward avg time 0.08315580654144288