import os
import glob
import dill
from torch.utils.data import \
    DataLoader

import bpe as bpe
import config
import time

import \
    gpt
from pipeline import \
    GetData

device = 'cuda'

current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = bpe.BPE.load(f'{ current_dir }/tokenizer.data')

with open('input_tokens.data', 'rb') as f:
    input_tokens = dill.load(f)


n = int(config.train_ratio * len(input_tokens)) # 90% train

train_token_ids = input_tokens[:n]
valid_token_ids = input_tokens[n:]

train_dataset = GetData(data=train_token_ids, seq_len=config.seq_len, device=device)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

valid_dataset = GetData(data=valid_token_ids, seq_len=config.seq_len, device=device)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

model = gpt.GPT(config.vocab_size, config.max_seq_len, config.emb_size, config.num_heads, config.head_size, config.num_layers, config.dropout, device=device)

model.tokenizer = tokenizer

model.fit(train_loader, valid_loader, config.num_epoch, config.learning_rate)
