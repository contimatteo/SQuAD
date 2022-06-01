# pylint: disable=unused-import,import-error
from typing import Any

import os

from tensorflow.keras.callbacks import EarlyStopping

import utils.env_setup
import utils.configs as Configs

from data import Dataset
from models import DRQA
from utils import LocalStorageManager, DataGenerator, FeaturesUtils, DataUtils

###

os.environ["WANDB_JOB_TYPE"] = "training"

LocalStorage = LocalStorageManager()

###


def __callbacks() -> list:
    callbacks = []

    callbacks.append(
        EarlyStopping(
            monitor='loss',
            patience=10,
            mode='min',
            min_delta=1e-4,
            restore_best_weights=True,
        )
    )

    return callbacks


def __fit(model, X, Y, passages_indexes, save_weights: bool, preload_weights: bool) -> Any:
    nn_epochs = Configs.NN_EPOCHS
    nn_batch = Configs.NN_BATCH_SIZE
    nn_callbacks = __callbacks()

    ### load weights
    nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    if preload_weights is True and nn_checkpoint_directory.is_file():
        model.load_weights(str(nn_checkpoint_directory))

    ### generator
    # generator = DataGenerator(X, Y, "size")
    generator = DataGenerator(X, Y, "passage", passages_indexes)

    ### train
    history = model.fit(
        generator.batches(),
        epochs=nn_epochs,
        batch_size=nn_batch,
        callbacks=nn_callbacks,
        steps_per_epoch=generator.steps_per_epoch
    )

    ### save weights
    if save_weights is True:
        nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
        model.save_weights(str(nn_checkpoint_directory), overwrite=True, save_format='h5')

    return history


###


def train():
    glove_matrix = Dataset.extract("glove")
    X, _ = FeaturesUtils.X_from_raw_features(Dataset.extract("features"))
    Y = FeaturesUtils.Y_from_raw_labels(Dataset.extract("labels"))
    _, _, _, passages_indexes = FeaturesUtils.QP_data_from_dataset(Dataset.extract("original"))

    ### save RAM memory
    Dataset.optimize_memory()

    #

    model = DRQA(glove_matrix)

    ### batches by passage
    _ = __fit(model, X, Y, passages_indexes, True, True)


###

if __name__ == "__main__":
    json_file_url = DataUtils.get_first_argv()
    assert isinstance(json_file_url, str)
    assert len(json_file_url) > 5
    assert ".json" in json_file_url
    Dataset.load(json_path=json_file_url)

    train()
