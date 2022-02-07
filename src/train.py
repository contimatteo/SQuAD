# pylint: disable=unused-import
from typing import Any, Tuple

import os
import numpy as np

from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from wandb.keras import WandbCallback

import utils.env_setup
import utils.configs as Configs

from data import get_data
from models import DRQA
from models.core import drqa_start_accuracy, drqa_end_accuracy, drqa_tot_accuracy
from models.core import drqa_start_mae, drqa_end_mae, drqa_tot_mae
from utils import XY_data_from_dataset, LocalStorageManager

###

os.environ["WANDB_JOB_TYPE"] = "training"

LocalStorage = LocalStorageManager()

###

_, dataset, glove_matrix, _ = get_data(300)


def __dataset() -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray]:
    x, y, _ = XY_data_from_dataset(dataset, Configs.NN_BATCH_SIZE * Configs.N_KFOLD_BUCKETS)

    return x, y


def __dataset_kfold(X, Y, indexes) -> list:
    return [el[indexes] for el in X], Y[indexes]


def __callbacks() -> list:
    callbacks = []

    callbacks.append(
        EarlyStopping(
            monitor='drqa_tot_crossentropy',
            patience=3,
            mode='min',
            min_delta=1e-3,
            restore_best_weights=True,
        )
    )

    if not Configs.WANDB_DISABLED:
        callbacks.append(WandbCallback())

    return callbacks


def __fit(model, X, Y, save_weights: bool) -> Any:
    nn_epochs = Configs.NN_EPOCHS
    nn_batch = Configs.NN_BATCH_SIZE
    nn_callbacks = __callbacks()

    history = model.fit(X, Y, epochs=nn_epochs, batch_size=nn_batch, callbacks=nn_callbacks)

    if save_weights is True:
        nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
        model.save_weights(str(nn_checkpoint_directory), overwrite=True, save_format='h5')

    return history


def __predict(model, X) -> np.ndarray:
    return model.predict(X)


def __evaluation(Y_true, Y_pred):
    start_accuracy = drqa_start_accuracy(Y_true, Y_pred).numpy()
    end_accuracy = drqa_end_accuracy(Y_true, Y_pred).numpy()
    tot_accuracy = drqa_tot_accuracy(Y_true, Y_pred).numpy()

    start_mae = drqa_start_mae(Y_true, Y_pred).numpy()
    end_mae = drqa_end_mae(Y_true, Y_pred).numpy()
    tot_mae = drqa_tot_mae(Y_true, Y_pred).numpy()

    return [start_accuracy, end_accuracy, tot_accuracy, start_mae, end_mae, tot_mae]


###


def train(X, Y):
    model = DRQA(glove_matrix)

    _ = __fit(model, X, Y, True)


def kfold_train(X, Y):
    metrics = []

    kf = KFold(n_splits=Configs.N_KFOLD_BUCKETS, shuffle=False)

    for train_indexes, test_indexes in kf.split(Y):
        model = DRQA(glove_matrix)

        ### split dataset in buckets
        X_train, Y_train = __dataset_kfold(X, Y, train_indexes)
        X_test, Y_test = __dataset_kfold(X, Y, test_indexes)

        ### train
        _ = __fit(model, X_train, Y_train, False)
        ### predict
        Y_test_pred = __predict(model, X_test)

        ### release Keras memory
        clear_session()

        ### evaluation
        metrics.append(__evaluation(Y_test, Y_test_pred))

    metrics = np.array(metrics)

    print()
    print("METRICS")
    print("[accuracy] start: ", metrics[:, 0].mean())
    print("[accuracy]   end: ", metrics[:, 1].mean())
    print("[accuracy] total: ", metrics[:, 2].mean())
    print("     [mae] start: ", metrics[:, 3].mean())
    print("     [mae]   end: ", metrics[:, 4].mean())
    print("     [mae] total: ", metrics[:, 5].mean())
    print()


###

if __name__ == "__main__":
    X, Y = __dataset()

    # kfold_train(X, Y)

    train(X, Y)
