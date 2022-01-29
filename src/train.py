import os
import random
import numpy as np

import utils.env_setup
import utils.configs as Configs

from wandb.keras import WandbCallback
from models import DRQA

###

os.environ["WANDB_JOB_TYPE"] = "training"

N_EXAMPLES = 500

###


def X_train() -> np.ndarray:
    Xq = np.random.random_sample((N_EXAMPLES, Configs.N_QUESTION_TOKENS))
    Xp = np.random.random_sample((N_EXAMPLES, Configs.N_PASSAGE_TOKENS))

    return [Xq, Xp]


def Y_train() -> np.ndarray:
    shape = (N_EXAMPLES, Configs.N_PASSAGE_TOKENS, 2)

    def _generator(n, k):
        """n numbers with sum k"""
        if n == 1:
            return [k]
        num = 0 if k == 0 else random.randint(1, k)
        return [num] + _generator(n - 1, k - num)

    def _Y_start_end():
        Y_start = np.array(_generator(Configs.N_PASSAGE_TOKENS, 100), dtype=np.float)
        Y_start /= 100
        Y_end = np.array(_generator(Configs.N_PASSAGE_TOKENS, 100), dtype=np.float)
        Y_end = Y_end[::-1] / 100
        # Y_start = np.zeros((Configs.N_PASSAGE_TOKENS, ))
        # Y_end = np.zeros((Configs.N_PASSAGE_TOKENS, ))
        # Y_start_index = random.randint(0, Configs.N_PASSAGE_TOKENS - 2)
        # Y_end_index = random.randint(Y_start_index + 1, Configs.N_PASSAGE_TOKENS - 1)
        # Y_start[Y_start_index] = 1
        # Y_end[Y_end_index] = 1

        return Y_start, Y_end

    Y = np.zeros(shape)
    for i in range(1):
        Y_start, Y_end = _Y_start_end()
        Y_true = np.array([Y_start, Y_end]).T
        Y[i] = Y_true

    return Y.astype(np.float)


def train():
    X = X_train()
    Y = Y_train()

    model = DRQA()

    model.fit(X, Y, epochs=5, batch_size=100, callbacks=[WandbCallback()])

    model.predict(X)


###

if __name__ == "__main__":
    train()
