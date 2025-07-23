import os
import glob
import bpe
import config

current_dir = os.path.dirname(os.path.abspath(__file__))

all_text = []

for file_path in glob.glob(f'{ current_dir }/data/*.*'):
    file = open(file_path, 'r', encoding='utf8')
    all_text.append(file.read())

all_text = '\n\n\n'.join(all_text)


tokenizer = bpe.BPE(config.vocab_size)

tokenizer.fit(all_text[:10000])

tokenizer.save(f'{ current_dir }/tokenizer.data')