from typing import Callable, Any

import tensorflow as tf
import numpy as np

import utils.configs as Configs

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.keras.activations import softmax
from tensorflow.keras.initializers import Constant

###


def GloveEmbeddings(input_length: int, initializer: np.ndarray) -> Callable[[Any], Any]:
    assert isinstance(input_length, int)

    i_dim = initializer.shape[0]  # size of the vocabulary
    o_dim = Configs.DIM_EMBEDDING  # dimension of the 'dense' embedding

    return Embedding(
        i_dim,
        o_dim,
        input_length=input_length,
        embeddings_initializer=Constant(initializer),
        trainable=False
    )


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
    nn1_units = Configs.N_PASSAGE_TOKENS + 1

    nn1_dense = Dense(
        1,
        activation="sigmoid",
        kernel_regularizer='l2',
        activity_regularizer='l2',
        bias_regularizer='l2',
        kernel_initializer='he_normal',
        bias_initializer='zeros'
    )
    nn2_start = Dense(
        nn1_units,
        activation="softmax",
        kernel_regularizer='l2',
        activity_regularizer='l2',
        bias_regularizer='l2',
        kernel_initializer='he_normal',
        bias_initializer='zeros'
    )
    nn2_end = Dense(
        nn1_units,
        activation="softmax",
        kernel_regularizer='l2',
        activity_regularizer='l2',
        bias_regularizer='l2',
        kernel_initializer='he_normal',
        bias_initializer='zeros'
    )

    def __nn1_add_complementar_bit(tensor: Any) -> Any:
        ### tensor shape --> (_, n_tokens)
        tensor_bit = nn1_dense(tensor)
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
        out_start = output[:, :, 0]
        ### --> (_, n_tokens)
        out_end = output[:, :, 1]
        ### --> (_, n_tokens)

        out_start = nn2_start(out_start)
        out_end = nn2_end(out_end)

        out_start = tf.expand_dims(out_start, axis=2)
        ### --> (_, n_tokens+1, 1)
        out_end = tf.expand_dims(out_end, axis=2)
        ### --> (_, n_tokens+1, 1)

        out_new = tf.concat([out_start, out_end], axis=2)
        ### --> (_, n_tokens+1, 2)

        return out_new

    return __nn1
    # return __nn2
