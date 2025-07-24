import os
import glob
import re
import dill
from torch.utils.data import \
    DataLoader

import bpe as bpe
import config
import time
import head_attention as mod
from head_attention import \
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

files = glob.glob(f'{ current_dir }/models/model_*.pt')

pattern = re.compile(r"model_(\d+)\.pt$")

epochs = [ pattern.search(file)[1] for file in files if pattern.search(file) != None ]

if len(epochs) == 0:
    print('no models found')
    exit(0)

epochs.sort(reverse=True)

model = mod.GPT.load(f'{ current_dir }/models/model_{ epochs[ 0 ] }.pt', device=device)
model.device = device
model.epoch = int(epochs[ 0 ]) + 1

model.fit(train_loader, valid_loader, config.num_epoch, config.learning_rate)
