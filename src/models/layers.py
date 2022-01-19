import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional, LSTM

###


class EmbeddingLayers():

    @staticmethod
    def glove(input_length):
        assert isinstance(input_length, int)

        i_dim = 100  # size of the vocabulary
        o_dim = 50  # dimension of the dense embedding

        # TODO: change params:
        #  - `trainable` --> `False`
        #  - `embeddings_initializer` --> `Constant(glove_matrix)`

        def _nn(x):
            x = Embedding(i_dim, o_dim, input_length=input_length, trainable=True)(x)
            x = Dropout(.3)(x)
            return x

        return _nn


###


class DenseLayers():

    @staticmethod
    def regularized():
        return Dense(5)


###


class RnnLayers():

    @staticmethod
    def drqa():
        units = 128
        initializer = 'glorot_uniform'

        def _lstm():
            return LSTM(units, dropout=.3, recurrent_initializer=initializer, return_sequences=True)

        def _nn(x):
            # x = Bidirectional(_lstm(), merge_mode="concat")(x)
            # x = Bidirectional(_lstm(), merge_mode="concat")(x)
            x = Bidirectional(_lstm(), merge_mode="concat")(x)
            return x

        return _nn


###


class AttentionLayers():

    @staticmethod
    def weighted_sum():

        def _nn(x):
            # --> (None, 20, 256)
            scores = Dense(1, use_bias=False)(x)  # 1 score foreach embedding input
            # --> (None, 20, 1)

            weights = Softmax(axis=1)(scores)
            # --> (None, 20, 1)

            # average
            x_weighted = weights * x
            # --> (None, 20, 1)

            # sum
            x_weighted_summed = tf.reduce_sum(x_weighted, axis=1)
            # --> (None, 20)

            return x_weighted_summed  #, x_weighted

        return _nn
