import \
    glob
import \
    os

current_dir = os.path.dirname(os.path.abspath(__file__))

corpus = []

for file_path in glob.glob(f'{ current_dir }/data/*.*'):
    file = open(file_path, 'r', encoding='utf8')
    corpus.append(file.read())

corpus = '\n\n\n'.join(corpus)

