from typing import Tuple

import os
import numpy as np

import utils.env_setup

from models import DRQA
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping

from data import get_data

###

os.environ["WANDB_JOB_TYPE"] = "training"

###


def __dataset() -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray]:
    X, Y = None, None

    _, data, glove, _ = get_data(300, debug=True)

    _, p_tokens, q_tokens, Y, p_pos, p_ner, p_tf, p_match = data

    X = [q_tokens, p_tokens, p_match, p_pos, p_ner, p_tf]

    return X, Y, glove


def __callbacks():
    return [
        WandbCallback(),
        EarlyStopping(
            monitor='loss',
            patience=3,
            mode='min',
            min_delta=1e-3,
            restore_best_weights=True,
        ),
        # ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_weights_only=True,
        #     monitor='val_accuracy',
        #     mode='max',
        #     save_best_only=True
        # )
    ]


###


def train():
    X, Y, glove = __dataset()

    model = DRQA(glove)

    model.fit(X, Y, epochs=10, batch_size=128, callbacks=__callbacks())

    model.predict(X)


###

if __name__ == "__main__":
    train()
