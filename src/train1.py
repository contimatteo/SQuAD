from pyexpat import model
from typing import Tuple

import os
from black import Any
import numpy as np

import utils.env_setup
import utils.configs as Configs

from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from models import DRQA
from data import get_data
from utils import XY_data_from_dataset
from models.core import metric

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


def extract_sub_dataset(X: Any, indexes: Any):
    X_new = []

    for el in X:
        el_new = np.array([el[i] for i in indexes], dtype=el.dtype)
        X_new.append(el_new)

    return X_new


def kfold_cross_validation(buckets=3):
    X, Y, glove = __dataset()

    assert buckets <= Y.shape[0]

    model = DRQA(glove)

    start = []
    end = []
    tot = []

    kf = KFold(n_splits=buckets, shuffle=False)

    for train_index, test_index in kf.split(Y):
        X_train = extract_sub_dataset(X, train_index)
        Y_train = Y[train_index]

        X_test = extract_sub_dataset(X, test_index)
        Y_test = Y[test_index]

        model.fit(
            X_train,
            Y_train,
            # epochs=Configs.EPOCHS,
            epochs=1,
            batch_size=Configs.BATCH_SIZE,
            callbacks=__callbacks()
        )

        Y_test_pred = model.predict(X_test)

        start_accuracy = metric.start_accuracy(Y_test, Y_test_pred)
        start.append(start_accuracy)
        end_accuracy = metric.end_accuracy(Y_test, Y_test_pred)
        end.append(end_accuracy)
        tot_accuracy = metric.tot_accuracy(Y_test, Y_test_pred)
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


def train():
    X, Y, glove = __dataset()

    model = DRQA(glove)

    model.fit(X, Y, epochs=Configs.EPOCHS, batch_size=Configs.BATCH_SIZE, callbacks=__callbacks())

    model.predict(X)


###

if __name__ == "__main__":
    # train()
    kfold_cross_validation()
