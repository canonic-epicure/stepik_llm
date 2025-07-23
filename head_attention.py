import math
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import \
    Dataset, \
    DataLoader
from typing import List

import tqdm

class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.emb_size: int = emb_size

        self.embeddings = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor):
        return self.embeddings(x)


class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self.max_seq_len: int = max_seq_len
        self.emb_size: int = emb_size

        self.embeddings = nn.Embedding(max_seq_len, emb_size)

    def forward(self, seq_len: int):
        seq = torch.arange(seq_len)

        return self.embeddings(seq.to(self.embeddings.weight.device))


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.emb_size: int = emb_size
        self.head_size: int = head_size
        self.max_seq_len = max_seq_len

        self.w_k = nn.Linear(self.emb_size, self.head_size)
        self.w_q = nn.Linear(self.emb_size, self.head_size)
        self.w_v = nn.Linear(self.emb_size, self.head_size)

        self.tril = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len))).to(self.w_k.weight.device)

    def forward(self, x: torch.Tensor):
        key: torch.Tensor = self.w_k(x)
        query: torch.Tensor = self.w_q(x)
        value: torch.Tensor = self.w_v(x)

        attention: torch.Tensor = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)

        attention_masked = attention.masked_fill(self.tril[0:attention.shape[-1], 0:attention.shape[-1]] == 0, float('-inf'))

        attention_soft = nn.functional.softmax(attention_masked, dim=-1)

        return attention_soft @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads: int = num_heads

        self.heads = nn.ModuleList([
            HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)
        ])

        self.out = nn.Linear(head_size * num_heads, emb_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([ head.forward(x) for head in self.heads ], dim=2)

        return self.dropout(self.out(out))


expand_size = 4

class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(emb_size, emb_size * expand_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(emb_size * expand_size, emb_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(
            self.linear2(
                self.relu(
                    self.linear1(x)
                )
            )
        )


class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        self.multi_head = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)

        self.feed_forward = FeedForward(emb_size, dropout)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor):
        out1 = self.norm1(self.multi_head(x) + x)

        return self.norm2(
            self.feed_forward(out1) + out1
        )


class GetData(Dataset):
    def __init__(self, data: List, seq_len: int, device='cpu'):
        super().__init__()

        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int):
        return (torch.tensor(self.data[ idx:idx + self.seq_len ]), torch.tensor(self.data[ idx + 1:idx + 1 + self.seq_len ]))


class GPT(nn.Module):
    def __init__(self, vocab_size:int, max_seq_len: int, emb_size: int, num_heads: int, head_size: int, num_layers: int, dropout: float = 0.1, device='cpu'):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.embeddings = TokenEmbeddings(vocab_size, emb_size)
        self.positional_embeddings = PositionalEmbeddings(max_seq_len, emb_size)

        self.dropout = nn.Dropout(dropout)
        self.decoders = nn.Sequential(*[ Decoder(num_heads, emb_size, head_size, max_seq_len, dropout) for _ in range(num_layers) ])
        self.linear = nn.Linear(emb_size, vocab_size)


    def forward(self, x: torch.Tensor):
        embs = self.dropout(self.embeddings(x) + self.positional_embeddings(x.shape[-1]))

        decoded = self.decoders(embs)

        return self.linear(decoded)


    def generate(self, x: torch.Tensor, max_new_tokens: int, do_sample: bool, temperature: float = 1.0, top_k: int = None, top_p: float = None):
        new_tokens = torch.zeros(x.shape[0], max_new_tokens).long()

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

        for e in tqdm.tqdm(range(num_epoch)):
            self.train()

            for inputs, targets in train_loader:
                res = self.forward(inputs)

                res_mod = res.reshape(res.shape[0] * res.shape[1], -1)
                targets_mod = targets.reshape(targets.shape[0] * targets.shape[1])

                self.train_loss = torch.nn.functional.cross_entropy(res_mod, targets_mod)

                print('train_loss=', torch.mean(self.train_loss))

                optimizer.zero_grad()
                self.train_loss.backward()
                optimizer.step()

            self.eval()

            with torch.no_grad():
                loss = []

                for inputs, targets in valid_loader:
                    res_val = self.forward(inputs).reshape(res.shape[0] * res.shape[1], -1)
                    targets_val = targets.reshape(targets.shape[0] * targets.shape[1])

                    valid_loss = torch.nn.functional.cross_entropy(res_val, targets_val)

                    loss.append(valid_loss)

            print('valid_loss=', torch.mean(valid_loss))

            self.save(f'./models/model_{e}.pt')

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
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
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model



if __name__ == "__main__":
    t = GPT(vocab_size=15, num_heads=2, emb_size=3, head_size=10, max_seq_len=40, num_layers=5)

    res = t.generate(
        torch.tensor([
            [1, 2, 3],
            [1, 2, 3]
        ]),
        10,
        True,
        0.9,
        None,
        0.4
        # False
    )

    print(res)
