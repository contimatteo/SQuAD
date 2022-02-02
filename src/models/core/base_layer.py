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


###


def EnhancedProbabilities() -> Callable[[Any], Any]:

    def __nn1(output: Any):
        out_start = output[:, :, 0]
        ### --> (batch_size, passage_len)
        out_end = output[:, :, 1]
        ### --> (batch_size, passage_len)

        out_bit_start = Dense(1, activation="sigmoid")(out_start)
        ### --> (batch_size,1)
        out_bit_end = Dense(1, activation="sigmoid")(out_end)
        ### --> (batch_size,1)

        out_bits = tf.concat([out_bit_start, out_bit_end], axis=1)
        ### --> (batch_size,2)

        out_bits = tf.expand_dims(out_bits, axis=1)
        ### --> (batch_size, 1, 2)

        output_new = tf.concat([output, out_bits], axis=1)
        ### --> (batch_size, passage_len +1, 2)

        output_new = softmax(output_new)
        ### --> (batch_size, passage_len +1, 2)

        return output_new

    def __nn2(output: Any) -> Any:
        units = Configs.N_PASSAGE_TOKENS + 1

        out_start = output[:, :, 0]
        ### --> (batch_size, passage_len)
        out_end = output[:, :, 1]
        ### --> (batch_size, passage_len)

        out_start = Dense(units, activation="softmax")(out_start)
        out_end = Dense(units, activation="softmax")(out_end)

        out_start = tf.expand_dims(out_start, axis=2)
        ### --> (batch_size, passage_len +1, 1)
        out_end = tf.expand_dims(out_end, axis=2)
        ### --> (batch_size, passage_len +1, 1)

        out_new = tf.concat([out_start, out_end], axis=2)
        ### --> (batch_size, passage_len +1, 2)

        return out_new

    return __nn1
