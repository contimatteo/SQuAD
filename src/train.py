from typing import Tuple

import os
import numpy as np

import utils.env_setup
import utils.configs as Configs

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

    X, Y = XY_data_from_dataset(data, Configs.BATCH_SIZE * 15)

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
    print("Starting Training")
    model = DRQA(glove)

    model.fit(X, Y, epochs=Configs.EPOCHS, batch_size=Configs.BATCH_SIZE, callbacks=__callbacks())

    model.predict(X)


###

if __name__ == "__main__":
    train()
