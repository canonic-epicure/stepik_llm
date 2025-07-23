import os
import glob
import bpe as bpe
import config
import time
import dill
from corpus import corpus

current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = bpe.BPE.load(f'{ current_dir }/tokenizer.data')

start = time.perf_counter()

tokens = tokenizer.encode(corpus[:1000])

with open('input_tokens.data', 'wb') as f:
    dill.dump(tokens, f)

end = time.perf_counter()

print(f'encoding done, time={ end - start }')

