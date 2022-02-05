# pylint: disable=unused-import
from typing import Tuple

import os
import numpy as np

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

    X, Y = XY_data_from_dataset(data, Configs.NN_BATCH_SIZE * Configs.N_KFOLD_BUCKETS)
    # X, Y = XY_data_from_dataset(data)

    return X, Y, glove


def __predict(model, X):
    nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    assert nn_checkpoint_directory.is_file()

    ### load weights
    model.load_weights(str(nn_checkpoint_directory))

    Y_pred = model.predict(X)

    return Y_pred


def __evaluation(Y_true, Y_pred):
    start_accuracy = drqa_accuracy_start(Y_true, Y_pred).numpy()
    end_accuracy = drqa_accuracy_end(Y_true, Y_pred).numpy()
    tot_accuracy = drqa_accuracy(Y_true, Y_pred).numpy()

    return [start_accuracy, end_accuracy, tot_accuracy]


###


def test():
    X, Y, glove = __dataset()

    model = DRQA(glove)

    Y_pred = __predict(model, X)

    metrics = __evaluation(Y, Y_pred)

    print()
    print("METRICS")
    print("[accuracy] start: ", metrics[0])
    print("[accuracy] end  : ", metrics[1])
    print("[accuracy] total: ", metrics[2])
    print()


###

if __name__ == "__main__":
    test()
