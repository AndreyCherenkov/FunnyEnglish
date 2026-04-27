from sklearn.datasets import fetch_20newsgroups
import re
from collections import Counter

from util.tokenizer import delete_stopwords_nltk

PAD = "<PAD>"
UNK = "<UNK>"

regex_email = r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
regex_link = r"https?://\S+|www\.\S+"
regex_digit = r"\b\d+[.,]?\d*\b"
regex_punctuation = r"[^\w\s]"
regex_html = r"<.*?>"

regex_email = re.compile(regex_email)
regex_link = re.compile(regex_link)
regex_digit = re.compile(regex_digit)
regex_punctuation = re.compile(regex_punctuation)
regex_html = re.compile(regex_html)

patterns = (
    regex_email,
    regex_link,
    regex_digit,
    regex_punctuation,
)


def load_data():
    train = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes')
    )
    test = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes')
    )
    print(train.target_names)
    return train, test


def tokenize(text):
    text = text.lower()

    text = regex_email.sub(" ", text)
    text = regex_link.sub(" ", text)
    text = regex_digit.sub(" ", text)

    text = text.replace("\n", " ")

    text = regex_html.sub(" ", text)
    text = regex_punctuation.sub(" ", text)

    tokens = text.split()

    return delete_stopwords_nltk(tokens)


def build_vocab(texts, vocab_size):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD: 0, UNK: 1}

    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)

    return vocab


def vectorize(texts, labels, vocab, max_len):
    result_x = []
    result_y = []

    for text, label in zip(texts, labels):
        tokens = tokenize(text)

        vec = []
        for w in tokens:
            vec.append(vocab.get(w, vocab[UNK]))

        vec = vec[:max_len]
        if len(vec) < max_len:
            vec += [vocab[PAD]] * (max_len - len(vec))

        result_x.append(vec)
        result_y.append(label)

    return result_x, result_y
