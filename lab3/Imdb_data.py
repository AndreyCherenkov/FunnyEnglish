from collections import Counter
from pathlib import Path

from util.tokenizer import delete_stopwords_nltk, tokenize

PAD = "<PAD>"
UNK = "<UNK>"
POS = "pos"
NEG = "neg"


class ImdbData:
    """
    Датасет IMDb для загрузки текстовых отзывов с метками классов.

    Структура данных предполагает папки:
        base_path/
            pos/ - положительные отзывы
            neg/ - отрицательные отзывы

    Каждый элемент датасета хранится как (file_path, label).
    """

    def __init__(self, base_path: str):
        self.samples = []
        base = Path(base_path)

        for label_name in [POS, NEG]:
            label = POS if label_name == POS else NEG

            for file in (base / label_name).glob("*.txt"):
                self.samples.append((file, label))

    def __len__(self) -> int:
        return len(self.samples)

    def get_samples(self):
        return self.samples


## Возвращает список с парами [токены исходного текста после удаления стоп-слов и стемминга] : метка отзыва
def process_texts(dataset: ImdbData) -> list[tuple[list[str], str]]:
    print("Processing texts...")
    data = []
    for text, label in dataset.get_samples():
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()
            tokenized_text = tokenize(text)
            tokenized_text = delete_stopwords_nltk(tokenized_text)
            # tokenized_text = stem_tokens_nltk(tokenized_text)
            data.append((tokenized_text, label))
            f.close()
    return data


def build_vocabulary(words: list[str], size: int = 25000) -> dict[str, int]:
    print("Building vocabulary")

    dictionary = {PAD: 0, UNK: 1}

    counter = Counter(words)

    most_common = counter.most_common(size - 2)

    for word, _ in most_common:
        dictionary[word] = len(dictionary)

    return dictionary


def vectorize_texts(
        data: list[tuple[list[str], str]],
        vocabulary: dict[str, int],
        max_len: int = 150
) -> list[tuple[list[int], int]]:
    result = []

    pad_idx = vocabulary[PAD]
    unk_idx = vocabulary[UNK]

    for text, label in data:
        vector = []
        for word in text:
            if word in vocabulary:
                vector.append(vocabulary[word])
            else:
                vector.append(unk_idx)

        vector = vector[:max_len]
        if len(vector) < max_len:
            vector += [pad_idx] * (max_len - len(vector))

        label_id = 1 if label == POS else 0

        result.append((vector, label_id))

    return result
