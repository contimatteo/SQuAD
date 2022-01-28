import os
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

    return np.random.randint(2, size=shape, dtype=int)


def train():
    X = X_train()
    Y = Y_train()

    model = DRQA()
    model.fit(X, Y, epochs=10, batch_size=50, callbacks=[WandbCallback()])


###

if __name__ == "__main__":
    train()
