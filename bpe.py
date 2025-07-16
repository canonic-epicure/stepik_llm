import torch

class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size: int = vocab_size
        self.id = 0
        self.id2token = {}
        self.token2id = {}

    def add_token(self, token: str):
        if token in self.token2id:
            print(f"Token {token} already exists")

        if self.id == self.vocab_size:
            print(f"Vocabulary size is {self.vocab_size}, cannot add token {token}")
            return

        id = self.id
        self.id += 1

        self.id2token[id] = token
        self.token2id[token] = id

    def fit(self, text: str):
        symbols = set()

        for c in text:
            symbols.add(c)

        symbols = list(symbols)
        symbols.sort()

        for s in symbols:
            self.add_token(s)



def main():
    bpe = BPE(30)
    bpe.fit('Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.')
    print(bpe.token2id.keys())


if __name__ == "__main__":
    main()