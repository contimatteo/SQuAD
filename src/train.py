# pylint: disable=unused-import
from typing import Tuple

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
from models.core import drqa_accuracy_start, drqa_accuracy_end, drqa_accuracy
from utils import XY_data_from_dataset, LocalStorageManager

###

os.environ["WANDB_JOB_TYPE"] = "training"

LocalStorage = LocalStorageManager()

###


def __dataset() -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray]:
    _, data, glove, _ = get_data(300)

    X, Y, _ = XY_data_from_dataset(data, Configs.NN_BATCH_SIZE * Configs.N_KFOLD_BUCKETS)
    # X, Y, _ = XY_data_from_dataset(data)

    return X, Y, glove


def __dataset_kfold(X, Y, indexes):
    return [el[indexes] for el in X], Y[indexes]


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


def __fit(model, X, Y, save_weights: bool):
    nn_epochs = Configs.NN_EPOCHS
    nn_batch = Configs.NN_BATCH_SIZE
    nn_callbacks = __callbacks()

    history = model.fit(X, Y, epochs=nn_epochs, batch_size=nn_batch, callbacks=nn_callbacks)

    if save_weights is True:
        nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
        model.save_weights(str(nn_checkpoint_directory), overwrite=True, save_format='h5')

    return history


def __predict(model, X):
    # nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    # assert nn_checkpoint_directory.is_file()
    # model.load_weights(str(nn_checkpoint_directory))

    Y_pred = model.predict(X)

    return Y_pred


def __evaluation(Y_true, Y_pred):
    start_accuracy = drqa_accuracy_start(Y_true, Y_pred).numpy()
    end_accuracy = drqa_accuracy_end(Y_true, Y_pred).numpy()
    tot_accuracy = drqa_accuracy(Y_true, Y_pred).numpy()

    return [start_accuracy, end_accuracy, tot_accuracy]


###


def train(X, Y, glove):

    model = DRQA(glove)

    _ = __fit(model, X, Y, True)


def kfold_train(X, Y, glove):
    metrics = []

    kf = KFold(n_splits=Configs.N_KFOLD_BUCKETS, shuffle=False)

    for train_indexes, test_indexes in kf.split(Y):
        model = DRQA(glove)

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
        eval_metrics = __evaluation(Y_test, Y_test_pred)
        metrics.append(eval_metrics)

    metrics = np.array(metrics)

    print()
    print("METRICS")
    print("[accuracy] start: ", metrics[:, 0].mean())
    print("[accuracy] end  : ", metrics[:, 1].mean())
    print("[accuracy] total: ", metrics[:, 2].mean())
    print()


###

if __name__ == "__main__":
    X, Y, glove = __dataset()

    # kfold_train(X, Y, glove)

    train(X, Y, glove)
