import math
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import \
    Dataset
from typing import List

from gpt import \
    GPT


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, device='cpu'):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.emb_size: int = emb_size

        self.embeddings = nn.Embedding(vocab_size, emb_size).to(device)

    def forward(self, x: torch.Tensor):
        return self.embeddings(x)


class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_size: int, device='cpu'):
        super().__init__()
        self.max_seq_len: int = max_seq_len
        self.emb_size: int = emb_size

        self.embeddings = nn.Embedding(max_seq_len, emb_size).to(device)

    def forward(self, seq_len: int):
        seq = torch.arange(seq_len)

        return self.embeddings(seq.to(self.embeddings.weight.device))


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int, device='cpu'):
        super().__init__()
        self.emb_size: int = emb_size
        self.head_size: int = head_size
        self.max_seq_len = max_seq_len

        self.w_k = nn.Linear(self.emb_size, self.head_size).to(device)
        self.w_q = nn.Linear(self.emb_size, self.head_size).to(device)
        self.w_v = nn.Linear(self.emb_size, self.head_size).to(device)

        self.tril = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len))).to(device)

    def forward(self, x: torch.Tensor):
        key: torch.Tensor = self.w_k(x)
        query: torch.Tensor = self.w_q(x)
        value: torch.Tensor = self.w_v(x)

        attention: torch.Tensor = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)

        attention_masked = attention.masked_fill(self.tril[0:attention.shape[-1], 0:attention.shape[-1]] == 0, float('-inf'))

        attention_soft = nn.functional.softmax(attention_masked, dim=-1)

        return attention_soft @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1, device='cpu'):
        super().__init__()
        self.num_heads: int = num_heads

        self.heads = nn.ModuleList([
            HeadAttention(emb_size, head_size, max_seq_len, device) for _ in range(num_heads)
        ])

        self.out = nn.Linear(head_size * num_heads, emb_size).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x: torch.Tensor):
        out = torch.cat([ head.forward(x) for head in self.heads ], dim=2)

        return self.dropout(self.out(out))


expand_size = 4

class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, device='cpu'):
        super().__init__()

        self.linear1 = nn.Linear(emb_size, emb_size * expand_size).to(device)
        self.relu = nn.ReLU().to(device)
        self.linear2 = nn.Linear(emb_size * expand_size, emb_size).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x: torch.Tensor):
        return self.dropout(
            self.linear2(
                self.relu(
                    self.linear1(x)
                )
            )
        )


class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1, device='cpu'):
        super().__init__()

        self.multi_head = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout, device)

        self.feed_forward = FeedForward(emb_size, dropout, device=device)

        self.norm1 = nn.LayerNorm(emb_size).to(device)
        self.norm2 = nn.LayerNorm(emb_size).to(device)

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
        return (torch.tensor(self.data[ idx:idx + self.seq_len ], device=self.device), torch.tensor(self.data[ idx + 1:idx + 1 + self.seq_len ], device=self.device))
