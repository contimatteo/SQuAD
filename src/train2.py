# pylint: disable=unused-import
from typing import Tuple

import os
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from wandb.keras import WandbCallback

import utils.env_setup
import utils.configs as Configs

from data import get_data
from models import DRQA
from models.core import drqa_accuracy_start, drqa_accuracy_end, drqa_accuracy
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
    # nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    # assert nn_checkpoint_directory.is_file()
    # model.load_weights(str(nn_checkpoint_directory))

    Y_pred = model.predict(X)

    return Y_pred


###


def train():
    X, Y_true, glove = __dataset()

    model = DRQA(glove)

    _ = __fit(model, X, Y_true)

    _ = __predict(model, X)


###


def extract_sub_dataset(X, indexes):
    X_new = []

    for el in X:
        el_new = np.array([el[i] for i in indexes], dtype=el.dtype)
        X_new.append(el_new)

    return X_new


def kfold_cross_validation():
    X, Y, glove = __dataset()

    model = DRQA(glove)

    start = []
    end = []
    tot = []

    buckets = 3
    kf = KFold(n_splits=buckets, shuffle=False)

    for train_index, test_index in kf.split(Y):
        X_train = extract_sub_dataset(X, train_index)
        Y_train = Y[train_index]

        X_test = extract_sub_dataset(X, test_index)
        Y_test = Y[test_index]

        model.fit(
            X_train,
            Y_train,
            epochs=Configs.NN_EPOCHS,
            batch_size=Configs.NN_BATCH_SIZE,
            callbacks=__callbacks()
        )

        Y_test_pred = model.predict(X_test)

        start_accuracy = drqa_accuracy_start(Y_test, Y_test_pred)
        start.append(start_accuracy)
        end_accuracy = drqa_accuracy_end(Y_test, Y_test_pred)
        end.append(end_accuracy)
        tot_accuracy = drqa_accuracy(Y_test, Y_test_pred)
        tot.append(tot_accuracy)

    start = np.array(start)
    end = np.array(end)
    tot = np.array(tot)

    print()
    print("Average test accuracy")
    print("start accuracy: ", start.mean())
    print("end   accuracy: ", end.mean())
    print("total accuracy: ", tot.mean())
    print()


###

if __name__ == "__main__":
    # train()
    kfold_cross_validation()
