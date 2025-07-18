from typing import List
import dill

class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size: int = vocab_size
        self.id = 0
        self.id2token = {}
        self.token2id = {}

        self.letter2tokens = {}

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

        letter = token[0]

        if letter not in self.letter2tokens:
            self.letter2tokens[ letter] = []

        self.letter2tokens[letter].append(token)

    def fit(self, text: str):
        symbols = set()

        for c in text:
            symbols.add(c)

        symbols = list(symbols)
        symbols.sort()

        for s in symbols:
            self.add_token(s)

        sequence = list(text)

        while len(self.id2token) < self.vocab_size:
            counts_dict = {}

            for i in range(len(sequence) - 1):
                token = sequence[i] + sequence[i + 1]

                if not token in counts_dict:
                    counts_dict[token] = 0

                counts_dict[token] += 1

            counts = list(counts_dict.items())
            max_entry = max(counts, key=lambda x: x[1])

            self.add_token(max_entry[0])

            i = 0
            while i < len(sequence) - 1:
                token = sequence[i] + sequence[i + 1]

                if token == max_entry[0]:
                    sequence[i] = max_entry[0]
                    sequence[ i + 1:i + 2 ] = []

                i += 1

        for letter in self.letter2tokens.keys():
            self.letter2tokens[letter].sort(key=len, reverse=True)

    def encode(self, text: str):
        sequence = list(text)
        ids = []

        i = 0
        while i < len(sequence):
            tokens = self.letter2tokens[sequence[i]]

            for token in tokens:
                if ''.join(sequence[i:i + len(token)]) == token:
                    ids.append(self.token2id[token])
                    i += len(token)

        return ids

    def decode(self, token_ids: List[int]):
        str = ''

        for id in token_ids:
            str += self.id2token[id]

        return str

    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            obj = dill.load(f)

        return obj

# if __name__ == "__main__":
#     bpe = BPE(30)
#     bpe.fit('Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.')
#     print(bpe.token2id.keys())
#
#     print(bpe.encode('Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'))
#
#     # bpe = BPE(31)
#     # bpe.fit('Однажды был случай в далёком Макао: макака коалу в какао макала, коала лениво какао лакала, макака макала, коала икала.')
#     # print(bpe.token2id.keys())
