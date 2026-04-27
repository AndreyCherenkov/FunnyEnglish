import numpy as np


def l2_normalize(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return X / norm