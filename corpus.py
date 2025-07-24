import \
    glob
import \
    os

current_dir = os.path.dirname(os.path.abspath(__file__))

corpus = []

paths = glob.glob(f'{ current_dir }/corpus/*.txt')

paths.sort()

for path in paths:
    with open(path, 'r', encoding='utf8') as file:
        corpus.append(file.read())

corpus = '\n\n\n'.join(corpus)

