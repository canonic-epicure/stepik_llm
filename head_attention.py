import math
import torch
import torch.nn as nn
import torch.nn.functional

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

        attention: torch.Tensor = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        attention_masked = attention.masked_fill(self.tril[0:attention.shape[-1], 0:attention.shape[-1]] == 0, float('-inf'))

        attention_soft = nn.functional.softmax(attention_masked, dim=-1)

        return torch.bmm(attention_soft, value)


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



# if __name__ == "__main__":
#     t = MultiHeadAttention(num_heads=2, emb_size=3, head_size=10, max_seq_len=32)
#
#     res = t.forward(
#         torch.tensor([[[ 1.9269,  1.4873,  0.9007],
#          [-2.1055,  0.6784, -1.2345],
#          [-0.0431, -1.6047, -0.7521],
#          [ 1.6487, -0.3925, -1.4036],
#          [-0.7279, -0.5594, -0.7688],
#          [ 0.7624,  1.6423, -0.1596]],
#
#         [[-0.4974,  0.4396,  0.3189],
#          [-0.4245,  0.3057, -0.7746],
#          [ 0.0349,  0.3211,  1.5736],
#          [-0.8455, -1.2742,  2.1228],
#          [-1.2347, -0.4879, -1.4181],
#          [ 0.8963,  0.0499,  2.2667]]])
#     )
#
#     print(res.shape)
#
#     # t = HeadAttention(emb_size=8, head_size=8, max_seq_len=24)
#     #
#     # x = torch.ones((1, 12, 8))
#     #
#     # t.forward(x)
#
