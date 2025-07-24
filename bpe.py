import llist
from typing import List
import dill
import re
import string

punc = re.compile(f"[{re.escape(string.punctuation)}]")
space = re.compile(f"\s")

class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size: int = vocab_size
        self.id = 0
        self.id2token = {}
        self.token2id = {}

        self.letter2tokens = {}

    def add_token(self, token: str, ignoreExisting = False):
        if not ignoreExisting and token in self.token2id:
            print(f"Token {token} already exists")
            return

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

    def all_tokens(self):
        return self.token2id.keys()

    def to_token(self, val1, val2):
        has_punc_1 = punc.search(val1) != None
        has_punc_2 = punc.search(val2) != None

        if not has_punc_1 and not has_punc_2:
            has_space_1 = space.search(val1) != None
            has_space_2 = space.search(val2) != None
            if not has_space_1 and not has_space_2:
                return val1 + val2

        return None

    def fit(self, text: str):
        symbols = set()

        for c in text:
            symbols.add(c)

        symbols = list(symbols)
        symbols.sort()

        for s in symbols:
            self.add_token(s)

        for char in range(ord('а'), ord('я')):
            self.add_token(chr(char), True)
            self.add_token(chr(char).upper(), True)

        for char in string.printable:
            self.add_token(char, True)

        sequence = llist.dllist(text)

        iter = 0

        while len(self.id2token) < self.vocab_size and sequence.size > 1:
            counts_dict = {}

            el = sequence.first

            while el != None:
                if el.next == None:
                    break

                token = self.to_token(el.value, el.next.value)

                if token != None:
                    if not token in counts_dict:
                        counts_dict[token] = 0

                    counts_dict[token] += 1

                el = el.next

            counts = list(counts_dict.items())
            max_entry = max(counts, key=lambda x: x[1])

            if max_entry[1] == 1:
                break

            self.add_token(max_entry[0])

            el = sequence.first

            while el != None:
                if el.next == None:
                    break

                token = self.to_token(el.value, el.next.value)

                if token == max_entry[0]:
                    el.value = max_entry[0]
                    sequence.remove(el.next)

                el = el.next

            iter += 1

            if iter % 1000 == 0:
                print(f"Iteration {iter}")

        for letter in self.letter2tokens.keys():
            self.letter2tokens[letter].sort(key=len, reverse=True)

    def encode(self, text: str) -> List[int]:
        sequence = list(text)
        ids = []

        i = 0
        while i < len(sequence):
            if sequence[i] in self.letter2tokens:
                tokens = self.letter2tokens[sequence[i]]

                for token in tokens:
                    if ''.join(sequence[i:i + len(token)]) == token:
                        ids.append(self.token2id[token])
                        i += len(token)
                        break
            else:
                i += 1

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
