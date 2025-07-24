import \
    os
import \
    bpe
import \
    config
import \
    gpt
import glob
import re
import torch

device = 'cuda'

current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = bpe.BPE.load(f'{ current_dir }/tokenizer.data')

files = glob.glob(f'{ current_dir }/models/model_*.pt')

pattern = re.compile(r"model_(\d+)\.pt$")

epochs = [ pattern.search(file)[1] for file in files if pattern.search(file) != None ]

if len(epochs) == 0:
    print('no models found')
    exit(0)

epochs.sort(reverse=True)

model = gpt.GPT.load(f'{ current_dir }/models/model_{ epochs[ 0 ] }.pt', device=device)

model = model.compile()

# total_params = sum(p.numel() for p in model.parameters())
# print("Total parameters: %.2fM" % (total_params / 1e6,))

input_text = 'Стояла хорошая погода'

token_ids = tokenizer.encode(input_text)
token_ids.extend([0] * (-len(token_ids) % config.seq_len))
token_ids = torch.tensor(token_ids).to(device)

output_text = model.generate(token_ids.reshape(config.batch_size, -1), config.max_seq_len, True, 0.9, 5, None)

decoded = tokenizer.decode(output_text.reshape(-1).tolist())

print(f'input_text={ input_text }')
print(f'output_text={ decoded }')