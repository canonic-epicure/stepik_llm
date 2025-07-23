import os
import glob
import bpe as bpe
import config
import time
import dill

current_dir = os.path.dirname(os.path.abspath(__file__))

all_text = []

for file_path in glob.glob(f'{ current_dir }/data/*.*'):
    file = open(file_path, 'r', encoding='utf8')
    all_text.append(file.read())

all_text = '\n\n\n'.join(all_text)


tokenizer = bpe.BPE(config.vocab_size)

start = time.perf_counter()

tokenizer.fit(all_text[:10000])

end = time.perf_counter()

tokenizer.save(f'{ current_dir }/tokenizer.data')

print(f'tokenizing done, time={ end - start }')
