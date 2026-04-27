from collections import Counter
from typing import Any

import numpy as np

from collections import Counter
from typing import Any


def build_vocabulary(
        tokenized_texts: list[list[str]],
        max_word: int = 5000,
        unknown_token: bool = False
) -> dict[Any, int]:
    counts = Counter()

    for tokens in tokenized_texts:
        counts.update(tokens)

    most_common = counts.most_common(max_word)

    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

    if unknown_token:
        vocab["<UNK>"] = len(vocab)

    return vocab

## for lab1
def build_bag_of_words(tokenized_texts: list[list[str]], vocab: dict[Any, int]) -> np.ndarray:
    vectors = np.zeros(
        (len(tokenized_texts), len(vocab)))  ## количество статей x длина вектора (соот. количеству слов словаря)
    for i, text in enumerate(tokenized_texts):
        for word in text:
            if word in vocab:
                index = vocab[word]
                vectors[i, index] += 1

    return vectors

## for lab2
def encode_texts(tokenized_texts: list[list[str]], vocab: dict[Any, int], max_len = 100) -> list[list[np.ndarray]]:
    encoded_texts = []
    for text in tokenized_texts:
        seq = []
        for word in text:
            if word in vocab:
                seq.append(vocab[word])
            else:
                seq.append(vocab["<UNK>"])

        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))
        encoded_texts.append(np.array(seq))
    return encoded_texts