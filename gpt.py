import \
    time
import \
    torch
import \
    tqdm
from torch import \
    nn as nn
from torch.utils.data import \
    DataLoader

from pipeline import \
    TokenEmbeddings, \
    PositionalEmbeddings, \
    Decoder


class GPT(nn.Module):
    def __init__(self, vocab_size:int, max_seq_len: int, emb_size: int, num_heads: int, head_size: int, num_layers: int, dropout: float = 0.1, device='cpu', epoch=1):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.embeddings = TokenEmbeddings(vocab_size, emb_size, device=device)
        self.positional_embeddings = PositionalEmbeddings(max_seq_len, emb_size, device=device)

        self.dropout = nn.Dropout(dropout).to(device)
        self.decoders = nn.Sequential(*[ Decoder(num_heads, emb_size, head_size, max_seq_len, dropout, device) for _ in range(num_layers) ])
        self.linear = nn.Linear(emb_size, vocab_size).to(device)

        self.epoch = epoch


    def forward(self, x: torch.Tensor):
        embs = self.dropout(self.embeddings(x) + self.positional_embeddings(x.shape[-1]))

        decoded = self.decoders(embs)

        return self.linear(decoded)


    def generate(self, x: torch.Tensor, max_new_tokens: int, do_sample: bool, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        new_tokens = torch.zeros(x.shape[0], max_new_tokens).long().to(self.device)

        for i in range(max_new_tokens):
            last = torch.cat([ x, new_tokens[:,:i] ], dim=-1)[:, -self.max_seq_len:]

            logits = self.forward(last)

            last_logs = logits[:, -1] / temperature

            if do_sample == False:
                probs = nn.functional.softmax(last_logs, dim=-1)
                max, indicies = torch.max(probs, -1)
                new_tokens[:, i] = indicies
            else:
                if top_k != None:
                    values, sorted_idx = torch.sort(last_logs, dim=-1, descending=True)

                    row_indices = torch.arange(last_logs.shape[0]).unsqueeze(1).expand(-1, last_logs.shape[1] - top_k)
                    last_logs[ row_indices, sorted_idx[ :, top_k: ] ] = float('-inf')

                if top_p != None:
                    probs = nn.functional.softmax(last_logs, dim=-1)
                    values, sorted_idx = torch.sort(probs, dim=-1, descending=True)

                    n, m = last_logs.shape
                    rows, cols = torch.meshgrid(torch.arange(n), torch.arange(m) )

                    cumsum = torch.cumsum(values, dim=-1)
                    cumsum[ :, 0 ] = 0

                    mask = cumsum >= top_p

                    last_logs[ rows[ mask ], sorted_idx[ mask ] ] = float('-inf')

                probs = nn.functional.softmax(last_logs, dim=-1)

                indicies = torch.multinomial(probs, 1)
                new_tokens[:, i] = indicies[:,0]

        return torch.cat([ x, new_tokens ], dim=-1)

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, num_epoch: int, learning_rate: float):
        self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for e in range(self.epoch, self.epoch + num_epoch):
            print(f'Starting epoch {e}')
            epoch_start = time.perf_counter()

            self.train()

            loss = []

            step = 0
            total_time = 0

            for inputs, targets in train_loader:
                start = time.perf_counter()

                res = self.forward(inputs)

                res_mod = res.reshape(res.shape[0] * res.shape[1], -1)
                targets_mod = targets.reshape(targets.shape[0] * targets.shape[1])

                train_loss = torch.nn.functional.cross_entropy(res_mod, targets_mod)
                loss.append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                end = time.perf_counter()

                step += 1
                step_time = end - start
                total_time += step_time

                print(f'\rtrain step={ step } done, from {len(train_loader)}, time={ step_time }s, loss={ train_loss.item() }, time remain={ (len(train_loader) - step) * (total_time / step) }s', end='', flush=True)

            print()

            train_loss = torch.mean(torch.tensor(loss)).item()

            self.eval()

            loss = []

            step = 0
            total_time = 0

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    start = time.perf_counter()

                    res = self.forward(inputs)

                    res_val = res.reshape(res.shape[0] * res.shape[1], -1)
                    targets_val = targets.reshape(targets.shape[0] * targets.shape[1])

                    valid_loss = torch.nn.functional.cross_entropy(res_val, targets_val)

                    loss.append(valid_loss.item())

                    end = time.perf_counter()

                    step += 1
                    step_time = end - start
                    total_time += step_time

                    print(f'\rvalid step={ step } done, from {len(valid_loader)}, time={ step_time }s, loss={ valid_loss.item() }, time remain={ (len(valid_loader) - step) * (total_time / step) }s', end='', flush=True)

            print(f'train_loss={ train_loss } valid_loss={ torch.mean(torch.tensor(loss)).item() }')

            epoch_end = time.perf_counter()
            print(f'Ending epoch {e}, time={ epoch_end - epoch_start }s')

            start = time.perf_counter()

            model_file = f'./models/model_{ str(e).rjust(3, "0") }.pt'

            self.save(model_file)

            end = time.perf_counter()

            print(f'\rEnding epoch {e}, time={ epoch_end - epoch_start }s, model saved to { model_file } within { end - start }s', end='', flush=True)

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers,
            'epoch': self.epoch,
        }, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers'],
            epoch=checkpoint['epoch'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
