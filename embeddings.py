import torch
import torch.nn as nn

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


# if __name__ == "__main__":
#     te = TokenEmbeddings(vocab_size=10, emb_size=5)
#
#     te.forward(
#         torch.IntTensor(
#             [
#                 [1, 2, 3, 4],
#                 [4, 3, 2, 1]
#             ]
#         )
#     )
#
