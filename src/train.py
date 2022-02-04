from typing import Tuple
import utils.env_setup

import os
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback

import utils.configs as Configs

from data import get_data
from models import DRQA
from utils import XY_data_from_dataset

###

os.environ["WANDB_JOB_TYPE"] = "training"

###


def __dataset() -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray]:
    _, data, glove, _ = get_data(300, debug=True)

    X, Y = XY_data_from_dataset(data, Configs.NN_BATCH_SIZE * 10)

    return X, Y, glove


def __callbacks():
    return [
        WandbCallback(),
        EarlyStopping(
            monitor='drqa_crossentropy',
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
    nn_epochs = Configs.NN_EPOCHS
    nn_batch = Configs.NN_BATCH_SIZE
    nn_callbacks = __callbacks()

    X, Y, glove = __dataset()

    model = DRQA(glove)

    model.fit(X, Y, epochs=nn_epochs, batch_size=nn_batch, callbacks=nn_callbacks)

    # model.predict(X)


###

if __name__ == "__main__":
    train()
