import csv
from tokenizer import tokenizer
import string
from collections import Counter
SOS_IDX = 0
PAD_IDX = 1 # PAD = EOS

tokenizer = tokenizer.RedditTokenizer()
punctuations = string.punctuation
punctuations = ''.join(set(punctuations) - set(',.'))

fpath = "/scratch/yn811/shortjokes.csv"

def tokenize(tokenizer, sent, punctuations):
    tokens = tokenizer.tokenize(sent)
    punc_cnt = sum((token in punctuations) for token in tokens)
    if punc_cnt > 0:
        return None
    return [token.lower() for token in tokens if (token not in punctuations)]

def tokenize_dataset(tokenizer, dataset, punctuations, gram=1):
    from tqdm import tqdm_notebook
    token_dataset = []
    all_tokens = []
    for sample in tqdm_notebook(dataset):
        tokens = tokenize(tokenizer, sample, punctuations)
        if tokens is None:
            continue
        if (len(tokens) <= 20) and (sum(len(w)<2 for w in tokens) <= len(tokens)/3):
            token_dataset.append(tokens)
            all_tokens.extend(tokens)
    return token_dataset, all_tokens

def build_vocab(all_tokens):
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(len(token_counter)))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2, 2+len(vocab)))) 
    id2token = ['<s>', '<pad>'] + id2token
    token2id['<pad>'] = PAD_IDX  
    token2id['<s>'] = SOS_IDX
    return token2id, id2token


def token2index_dataset(tokens_data):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] for token in tokens]
        indices_data.append(index_list)
    return indices_data

# ! pip install git+https://github.com/erikavaris/tokenizer.git
jokes = []
with open(fpath) as f:
    reader = csv.reader(f) 
    next(reader, None)
    for row in reader:
        jokes.append(row[1])

token_dataset, all_tokens = tokenize_dataset(tokenizer, jokes, punctuations)
token2id, id2token = build_vocab(all_tokens)
idx_data = token2index_dataset(token_dataset)
import pickle as pkl
pkl.dump([idx_data, token_dataset, token2id, id2token], open("short_jokes.pkl", "wb"))
