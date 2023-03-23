import os
import sys
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import get_peft_model, TaskType, LoraConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16')
pad_token_id = tokenizer.pad_token_id
device = 'cuda'


def batch_process(batch):
    # prompt format
    query_text = ["내용: " + text + "\n요약: " for text in batch['text']]
    target_text = batch['summary']

    # tokenize
    query = tokenizer(query_text)
    target = tokenizer(target_text)

    input_ids = [q + t + [tokenizer.eos_token_id] for q, t in zip(query['input_ids'], target['input_ids'])]
    attention_mask = [q + t + [1] for q, t in zip(query['attention_mask'], target['attention_mask'])]
    labels = [[-100] * len(q) + t + [tokenizer.eos_token_id] for q, t in zip(query['input_ids'], target['input_ids'])]

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def get_model():
    # base model
    base_model = AutoModelForCausalLM.from_pretrained(
        'kakaobrain/kogpt',
        revision='KoGPT6B-ryan1.5b-float16',
        torch_dtype=torch.float16,
        device_map='auto'
    )

    # peft model
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj']
    )

    peft_model = get_peft_model(base_model, peft_cfg)

    return peft_model


def get_dataset(path):
    train_df = pd.read_csv(path)
    train_ds = Dataset.from_pandas(train_df)

    train_ds = train_ds.map(
        batch_process,
        # remove_columns=['id', 'text', 'summary'],
        batched=True,
        batch_size=1000
    )
    return train_ds


def get_dataloader(path):
    dataset = get_dataset(path)
    dataset = dataset.map(
        batch_process,
        remove_columns=['id', 'text', 'summary'],
        batched=True,
        batch_size=512
    )
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=os.cpu_count() // 2, collate_fn=collate_fn
    )
    return dataloader


def left_pad(seq, value, max_len):
    return [value] * (max_len - len(seq)) + seq


def collate_fn(batch):
    length = max(len(row['input_ids']) for row in batch)

    input_ids = [left_pad(row['input_ids'], pad_token_id, length) for row in batch]
    attention_mask = [left_pad(row['attention_mask'], 0, length) for row in batch]
    labels = [left_pad(row['input_ids'], -100, length) for row in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


TOTAL_LENGTH = 80
last = time.time()
begin = last
term_width = 120


def progress_bar(current, total, msg=None):
    global last, begin
    if current == 0:
        begin = time.time()

    current_len = int(TOTAL_LENGTH * current / total)
    rest_len = int(TOTAL_LENGTH - current_len) - 1

    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last
    last = cur_time
    tot_time = cur_time - begin

    msg_bar = [f"  Step: {format_time(step_time)} | Total: {format_time(tot_time)}"]
    if msg:
        msg_bar.append(' | ' + msg)

    msg = ''.join(msg_bar)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
