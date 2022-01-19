import utils.warnings

import numpy as np

from models import DrQA

###

N_EXAMPLES = 20


def X_train() -> np.ndarray:
    N_QUESTION_TOKENS = 5
    Xq = np.random.rand(N_EXAMPLES, N_QUESTION_TOKENS)

    N_PASSAGE_TOKENS = 10
    Xp = np.random.rand(N_EXAMPLES, N_PASSAGE_TOKENS)

    return [Xq, Xp]


def Y_train() -> np.ndarray:
    return np.random.rand(N_EXAMPLES, )


def train():
    X = X_train()
    Y = Y_train()

    model = DrQA()
    model.fit(X, Y, epochs=10)


###

if __name__ == "__main__":
    train()
