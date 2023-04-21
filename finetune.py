import torch
import os
import json
from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast
from utils import load_lora_config

checkpoint = "THUDM/chatglm-6b"
revision = "096f3de6b4959ce38bef7bb05f3129c931a3084e"
model = AutoModel.from_pretrained(checkpoint, revision = revision, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision = revision, trust_remote_code=True)

model = load_lora_config(model)

bos = 150004
eop = 150005
pad = 20003
mask = 150000
gmask = 150001
device = "cuda"
max_src_length = 200
max_dst_length = 500

PROMPT_PATTERN = "问：{}"
SEP_PATTERN = "\n答： "

def create_prompt(question):
    return PROMPT_PATTERN.format(question), SEP_PATTERN


def create_prompt_ids(tokenizer, question, max_src_length):
    prompt, sep = create_prompt(question)
    sep_ids = tokenizer.encode(
        sep,
        add_special_tokens = True
    )
    sep_len = len(sep_ids)
    special_tokens_num = 2
    prompt_ids = tokenizer.encode(
        prompt,
        max_length = max_src_length - (sep_len - special_tokens_num),
        truncation = True,
        add_special_tokens = False
    )

    return prompt_ids + sep_ids


def create_inputs_and_labels(tokenizer, question, answer, device):
    prompt = create_prompt_ids(tokenizer, question, max_src_length)
    completion = tokenizer.encode(
        answer,
        max_length = max_dst_length,
        truncation = True,
        add_special_tokens = False
    )

    inputs = prompt + completion + [eop]
    labels = [-100] * len(prompt) + completion + [eop]

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return inputs, labels

def get_attention_mask(tokenizer, input_ids, device):
    seq = input_ids.tolist()
    context_len = seq.index(bos)
    seq_len = len(seq)
    attention_mask = torch.ones((seq_len, seq_len), device=device)
    attention_mask.tril_()
    attention_mask[..., :context_len] = 1
    attention_mask.unsqueeze_(0)
    attention_mask = (attention_mask < 0.5).bool()
    return attention_mask


def get_position_ids(tokenizer, input_ids, device, position_encoding_2d=True):
    seq = input_ids.tolist()
    context_len = seq.index(bos)
    seq_len = len(seq)

    mask_token = mask if mask in seq else gmask
    use_gmask = False if mask in seq else gmask

    mask_position = seq.index(mask_token)

    if position_encoding_2d:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position
        block_position_ids = torch.cat((
            torch.zeros(context_len, dtype=torch.long, device=device),
            torch.arange(seq_len - context_len, dtype=torch.long, device=device) + 1
        ))
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        if not use_gmask:
            position_ids[context_len:] = mask_position

    return position_ids

from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer


    def __getitem__(self, index):
        item_data = self.data[index]
        tokenizer = self.tokenizer
        input_ids, labels = create_inputs_and_labels(
            tokenizer,
            device=device,
            **item_data
        )

        attention_mask = get_attention_mask(tokenizer, input_ids, device)
        position_ids = get_position_ids(tokenizer, input_ids, device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }


    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    position_ids = []

    for obj in batch:
        input_ids.append(obj['input_ids'])
        labels.append(obj['labels'])
        attention_mask.append(obj['attention_mask'])
        position_ids.append(obj['position_ids'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
        'position_ids':torch.stack(position_ids)
    }


train_data = []
with open("data/qa.json", encoding="utf-8") as file:
    train_data = json.load(file)

from transformers import TrainingArguments, Trainer
model.to(device)

training_args = TrainingArguments(
    "output",
    fp16 =True,
    save_steps = 500,
    save_total_limit = 3,
    gradient_accumulation_steps=1,
    per_device_train_batch_size = 1,
    learning_rate = 1e-4,
    max_steps=1500,
    logging_steps=50,
    remove_unused_columns=False,
    seed=0,
    data_seed=0,
    group_by_length=False,
    dataloader_pin_memory=False
)

class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            labels=inputs["labels"],
        ).loss


train_dataset = QADataset(train_data, tokenizer=tokenizer)
trainer = ModifiedTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=collate_fn,
    tokenizer=tokenizer
)



trainer.train()

def save_tuned_parameters(model, path):
    saved_params = {
        k: v.to(device)
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)

save_tuned_parameters(model, os.path.join("output", "chatglm-6b-lora.pt"))
