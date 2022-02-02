from typing import Tuple

import os
import numpy as np

import utils.env_setup

from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback
from models import DRQA
from data import get_data
from utils import XY_data_from_dataset

###

os.environ["WANDB_JOB_TYPE"] = "training"

###


def __dataset() -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray]:
    _, data, glove, _ = get_data(300, debug=True)

    X, Y = XY_data_from_dataset(data)

    # print()
    # print()
    # print("q_tokens = ", X[0].shape)
    # print("p_tokens = ", X[1].shape)
    # print("p_match = ", X[2].shape)
    # print("p_pos = ", X[3].shape)
    # print("p_ner = ", X[4].shape)
    # print("p_tf = ", X[5].shape)
    # print()
    # print()

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
