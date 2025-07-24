import bpe
import os
import glob
import config
import time
from corpus import corpus

current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = bpe.BPE(config.vocab_size)

start = time.perf_counter()

tokenizer.fit(corpus[:50000])

end = time.perf_counter()

tokenizer.save(f'{ current_dir }/tokenizer.data')

print(f'tokenizing done, time={ end - start }')
