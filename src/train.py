import os
from random import randint
import numpy as np

import utils.env_setup
import utils.configs as Configs

from wandb.keras import WandbCallback
from models import DRQA

###

os.environ["WANDB_JOB_TYPE"] = "training"

N_ROWS = 500

###


def X_train() -> np.ndarray:
    ### [Q] tokens
    Q_tokens = np.random.random_sample((N_ROWS, Configs.N_QUESTION_TOKENS))

    ### [P] tokens
    P_tokens = np.random.random_sample((N_ROWS, Configs.N_PASSAGE_TOKENS))

    ### [P] exact match
    P_match = np.zeros((N_ROWS, Configs.N_PASSAGE_TOKENS, Configs.DIM_EXACT_MATCH)).astype(np.float)

    ### [P] POS tags
    P_pos = np.zeros((N_ROWS, Configs.N_PASSAGE_TOKENS, Configs.N_POS_CLASSES)).astype(np.float)

    ### [P] NER annotations
    P_ner = np.random.random_sample((N_ROWS, Configs.N_PASSAGE_TOKENS, Configs.N_NER_CLASSES))

    ### [P] TF
    P_tf = np.zeros((N_ROWS, Configs.N_PASSAGE_TOKENS, Configs.DIM_TOKEN_TF)).astype(np.float)

    return [Q_tokens, P_tokens, P_match, P_pos, P_ner, P_tf]


def Y_train() -> np.ndarray:
    shape = (N_ROWS, Configs.N_PASSAGE_TOKENS, 2)

    def _generator(n, k):
        """n numbers with sum k"""
        if n == 1:
            return [k]
        num = 0 if k == 0 else randint(1, k)
        return [num] + _generator(n - 1, k - num)

    def _Y_start_end():
        Y_start = np.array(_generator(Configs.N_PASSAGE_TOKENS, 100), dtype=np.float)
        Y_start /= 100
        Y_end = np.array(_generator(Configs.N_PASSAGE_TOKENS, 100), dtype=np.float)
        Y_end = Y_end[::-1] / 100
        # Y_start = np.zeros((Configs.N_PASSAGE_TOKENS, ))
        # Y_end = np.zeros((Configs.N_PASSAGE_TOKENS, ))
        # Y_start_index = randint(0, Configs.N_PASSAGE_TOKENS - 2)
        # Y_end_index = randint(Y_start_index + 1, Configs.N_PASSAGE_TOKENS - 1)
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
