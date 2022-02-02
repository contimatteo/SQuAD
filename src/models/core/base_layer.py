from typing import Callable, Any, List

import tensorflow as tf
import numpy as np

import utils.configs as Configs

from tensorflow.keras.layers import Dense, Dropout, Dot
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.keras.activations import softmax
from tensorflow.keras.initializers import Constant

###


def GloveEmbeddings(input_length: int, initializer: np.ndarray) -> Callable[[Any], Any]:
    assert isinstance(input_length, int)

    i_dim = initializer.shape[0]  # size of the vocabulary
    o_dim = Configs.DIM_EMBEDDING  # dimension of the 'dense' embedding

    def _nn(inp: Any) -> Any:
        x = Embedding(
            i_dim,
            o_dim,
            input_length=input_length,
            embeddings_initializer=Constant(initializer),
            trainable=False
        )(inp)
        x = Dropout(.3)(x)
        return x

    return _nn


###


def DrqaRnn() -> Callable[[Any], Any]:
    units = 128
    initializer = 'glorot_uniform'

    def _lstm() -> LSTM:
        return LSTM(units, dropout=.3, recurrent_initializer=initializer, return_sequences=True)

    def _nn(inp: Any) -> Any:
        x = Bidirectional(_lstm(), merge_mode="concat")(inp)
        x = Bidirectional(_lstm(), merge_mode="concat")(x)
        x = Bidirectional(_lstm(), merge_mode="concat")(x)
        return x

    return _nn
