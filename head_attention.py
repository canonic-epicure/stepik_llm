import math
import torch
import torch.nn as nn
import torch.nn.functional

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
        return self.embeddings(torch.arange(seq_len))


class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.emb_size: int = emb_size
        self.head_size: int = head_size
        self.max_seq_len = max_seq_len

        self.w_k = nn.Linear(self.emb_size, self.head_size)
        self.w_q = nn.Linear(self.emb_size, self.head_size)
        self.w_v = nn.Linear(self.emb_size, self.head_size)

        self.tril = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len)))

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


    def generate(self, x: torch.Tensor, max_new_tokens: int):
        new_tokens = torch.zeros(x.shape[0], max_new_tokens).long()

        for i in range(max_new_tokens):
            last = torch.cat([ x, new_tokens[:,:i] ], dim=-1)[:, -self.max_seq_len:]

            logits = self.forward(last)

            # probs = nn.functional.softmax(logits[:, -1], dim=-1)

            max, indicies = torch.max(logits[:, -1], -1)

            new_tokens[:, i] = indicies

        return torch.cat([ x, new_tokens ], dim=-1)

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
        10
    )

    print(res)
