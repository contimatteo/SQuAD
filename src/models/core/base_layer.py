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

    def __nn1_add_complementar_bit(tensor: Any) -> Any:
        ### tensor shape --> (_, n_tokens)

        tensor_bit = Dense(1, activation="sigmoid")(tensor)
        ### --> (_, 1)
        tensor_new = tf.concat([tensor, tensor_bit], axis=1)
        ### --> (_, n_tokens+1)
        tensor_new = softmax(tensor_new)
        ### --> (_, n_tokens+1)
        tensor_new = tf.expand_dims(tensor_new, axis=2)
        ### --> (_, n_tokens+1, 1)

        return tensor_new

    def __nn1(output: Any):
        out_start = output[:, :, 0]
        ### --> (_, n_tokens)
        out_end = output[:, :, 1]
        ### --> (_, n_tokens)

        out_start = __nn1_add_complementar_bit(out_start)
        ### --> (_, n_tokens+1, 1)
        out_end = __nn1_add_complementar_bit(out_end)
        ### --> (_, n_tokens+1, 1)

        output_new = tf.concat([out_start, out_end], axis=2)
        ### --> (_, n_tokens+1, 2)

        return output_new

    def __nn2(output: Any) -> Any:
        units = Configs.N_PASSAGE_TOKENS + 1

        out_start = output[:, :, 0]
        ### --> (_, n_tokens)
        out_end = output[:, :, 1]
        ### --> (_, n_tokens)

        out_start = Dense(units, activation="softmax")(out_start)
        out_end = Dense(units, activation="softmax")(out_end)

        out_start = tf.expand_dims(out_start, axis=2)
        ### --> (_, n_tokens+1, 1)
        out_end = tf.expand_dims(out_end, axis=2)
        ### --> (_, n_tokens+1, 1)

        out_new = tf.concat([out_start, out_end], axis=2)
        ### --> (_, n_tokens+1, 2)

        return out_new

    return __nn1
