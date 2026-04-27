import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.utils import Bunch

import re

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


def preprocess_text(text):
    text = text.lower()

    text = regex_email.sub(" ", text)
    text = regex_link.sub(" ", text)
    text = regex_digit.sub(" ", text)

    text = text.replace("\n", " ")

    text = regex_html.sub(" ", text)
    text = regex_punctuation.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return preprocess_text(text).split()


def tokenize_texts(dataset: Bunch) -> list[list[str]]:
    return [tokenize(text) for text in dataset.data]


##nltk block
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


def delete_stopwords_nltk(tokens: list[str]) -> list[str]:
    sw = nltk.corpus.stopwords.words('english')
    return [w for w in tokens if w not in sw]


def stem_tokens_nltk(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens_nltk(tokens: list[str]) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def process(
        tokens: list[list[str]],
        use_stopwords=False,
        use_lemmatize=False,
        use_stem=False
) -> list[list[str]]:
    processed = []

    for doc in tokens:
        new_doc = doc.copy()

        if use_stopwords:
            new_doc = delete_stopwords_nltk(new_doc)

        if use_lemmatize:
            new_doc = lemmatize_tokens_nltk(new_doc)

        if use_stem:
            new_doc = stem_tokens_nltk(new_doc)

        processed.append(new_doc)

    return processed
