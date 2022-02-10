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

from data import get_data, load_data, delete_data
from models import DRQA
from models.core import drqa_start_accuracy_metric, drqa_end_accuracy_metric, drqa_accuracy_metric
from utils import LocalStorageManager
from utils import X_data_from_dataset, Y_data_from_dataset
from utils.generator import Generator
from utils.memory_usage import memory_usage

import tensorflow as tf
###

os.environ["WANDB_JOB_TYPE"] = "training"

LocalStorage = LocalStorageManager()

N_ROWS_SUBSET = None  # `None` for all rows :)

###


def __dataset_kfold(X, Y, indexes) -> list:
    return [el[indexes] for el in X], Y[indexes]


def __callbacks() -> list:
    callbacks = []

    # callbacks.append(
    #     EarlyStopping(
    #         monitor='loss',
    #         patience=5,
    #         mode='min',
    #         min_delta=1e-4,
    #         restore_best_weights=True,
    #     )
    # )
    # if not Configs.WANDB_DISABLED:
    #     callbacks.append(WandbCallback())

    return callbacks


def __fit(model, X, Y, save_weights: bool, preload_weights: bool) -> Any:
    nn_epochs = Configs.NN_EPOCHS
    nn_batch = Configs.NN_BATCH_SIZE
    nn_callbacks = __callbacks()

    ### generator
    gen = Generator(X, Y)
    dataset = gen.generate_dynamic_batches()
    steps_per_epoch = gen.get_steps_per_epoch()

    ### load weights
    nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    if preload_weights is True and nn_checkpoint_directory.is_file():
        model.load_weights(str(nn_checkpoint_directory))

    ### train
    history = model.fit(
        dataset,
        epochs=nn_epochs,
        batch_size=nn_batch,
        callbacks=nn_callbacks,
        steps_per_epoch=steps_per_epoch,
        max_queue_size=1
    )

    if save_weights is True:
        nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
        model.save_weights(str(nn_checkpoint_directory), overwrite=True, save_format='h5')

    return history


def __predict(model, X) -> np.ndarray:
    return model.predict(X)


def __evaluation(Y_true, Y_pred):
    start_accuracy = drqa_start_accuracy_metric(Y_true, Y_pred).numpy()
    end_accuracy = drqa_end_accuracy_metric(Y_true, Y_pred).numpy()
    tot_accuracy = drqa_accuracy_metric(Y_true, Y_pred).numpy()

    # start_mae = drqa_start_mae(Y_true, Y_pred).numpy()
    # end_mae = drqa_end_mae(Y_true, Y_pred).numpy()
    # tot_mae = drqa_tot_mae(Y_true, Y_pred).numpy()
    # return [start_accuracy, end_accuracy, tot_accuracy, start_mae, end_mae, tot_mae]

    return [start_accuracy, end_accuracy, tot_accuracy]


###


def train():
    X, _ = X_data_from_dataset(get_data("features"), N_ROWS_SUBSET)
    Y = Y_data_from_dataset(get_data("labels"), N_ROWS_SUBSET)
    glove_matrix = get_data("glove")

    print("After numpy")
    memory_usage()
    delete_data()
    print("After deleted data")
    memory_usage()

    model = DRQA(glove_matrix)

    _ = __fit(model, X, Y, True, True)


def kfold_train():
    X, _ = X_data_from_dataset(get_data("features"), N_ROWS_SUBSET)
    Y = Y_data_from_dataset(get_data("labels"), N_ROWS_SUBSET)
    glove_matrix = get_data("glove")

    metrics = []

    kf = KFold(n_splits=Configs.N_KFOLD_BUCKETS, shuffle=False)

    for train_indexes, test_indexes in kf.split(Y):
        model = DRQA(glove_matrix)

        ### split dataset in buckets
        X_train, Y_train = __dataset_kfold(X, Y, train_indexes)
        X_test, Y_test = __dataset_kfold(X, Y, test_indexes)

        ### train
        _ = __fit(model, X_train, Y_train, False, False)
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
    # load_data(json_path="./data/raw/train.v1.json")
    load_data()

    print("After preprocessing")
    memory_usage()

    # kfold_train()

    train()
