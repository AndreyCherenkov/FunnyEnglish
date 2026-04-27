from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch


def load_train_dataset_20newsgroups() -> Bunch:
    data = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes')
    )
    print(data.target_names)
    return data


def load_test_dataset_20newsgroups() -> Bunch:
    data = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes')
    )
    return data
