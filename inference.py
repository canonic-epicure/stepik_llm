import \
    os
import \
    bpe as bpe
import \
    config
import \
    gpt

import torch

device = 'cuda'

current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = bpe.BPE.load(f'{ current_dir }/tokenizer.data')

model = gpt.GPT.load(f'{ current_dir }/models/model_16.pt', device=device)
model.device = device


# total_params = sum(p.numel() for p in model.parameters())
# print("Total parameters: %.2fM" % (total_params / 1e6,))

input_text = 'Стояла хорошая погода'

token_ids = torch.tensor(tokenizer.encode(input_text)).to(device)

output_text = model.generate(token_ids.reshape(config.batch_size, -1), config.max_seq_len, True, 0.9, 5, None)

decoded = tokenizer.decode(output_text.reshape(-1).tolist())

print(f'input_text={ input_text }')
print(f'output_text={ decoded }')