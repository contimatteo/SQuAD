# pylint: disable=unused-import
from typing import Tuple

import os
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback

import utils.env_setup
import utils.configs as Configs

from data import get_data
from models import DRQA
from utils import XY_data_from_dataset, LocalStorageManager

###

os.environ["WANDB_JOB_TYPE"] = "training"

LocalStorage = LocalStorageManager()

###


def __dataset() -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray]:
    _, data, glove, _ = get_data(300, debug=True)

    X, Y = XY_data_from_dataset(data, Configs.NN_BATCH_SIZE * 10)

    return X, Y, glove


def __callbacks():
    callbacks = []

    callbacks.append(
        EarlyStopping(
            monitor='drqa_loss',
            patience=3,
            mode='min',
            min_delta=1e-3,
            restore_best_weights=True,
        )
    )

    if not Configs.WANDB_DISABLED:
        callbacks.append(WandbCallback())

    return callbacks


def __fit(model, X, Y):
    nn_epochs = Configs.NN_EPOCHS
    nn_batch = Configs.NN_BATCH_SIZE
    nn_callbacks = __callbacks()
    nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)

    history = model.fit(X, Y, epochs=nn_epochs, batch_size=nn_batch, callbacks=nn_callbacks)

    model.save_weights(str(nn_checkpoint_directory), overwrite=True, save_format=None, options=None)

    return history


def __predict(model, X):
    nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    assert nn_checkpoint_directory.is_file()

    model.load_weights(str(nn_checkpoint_directory))

    Y_pred = model.predict(X)

    return Y_pred


###


def train():
    X, Y_true, glove = __dataset()

    model = DRQA(glove)

    _ = __fit(model, X, Y_true)

    _ = __predict(model, X)


###

if __name__ == "__main__":
    train()
