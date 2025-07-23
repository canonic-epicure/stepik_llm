import \
    os
import \
    bpe as bpe
import \
    config
import \
    head_attention as mod

device = 'cuda'

current_dir = os.path.dirname(os.path.abspath(__file__))

tokenizer = bpe.BPE.load(f'{ current_dir }./tokenizer.data')

model = mod.GPT.load(f'{ current_dir }/models/model_19.pt', device=device)

input_text = 'Стояла хорошая погода'

token_ids = tokenizer.encode(input_text)

output_text = model.generate(token_ids, config.max_seq_len, True, 0.9, 5, None)

decoded = tokenizer.decode(output_text)

print(f'input_text={input_text}')
print(f'output_text={output_text}')