from typing import Callable, Any, List, Tuple

import tensorflow as tf

from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional, LSTM

###


class EmbeddingLayers():

    @staticmethod
    def glove(input_length: int) -> Callable[[Any], Any]:
        assert isinstance(input_length, int)

        i_dim = 100  # size of the vocabulary
        o_dim = 50  # dimension of the dense embedding

        # TODO: change params:
        #  - `trainable` --> `False`
        #  - `embeddings_initializer` --> `Constant(glove_matrix)`

        def _nn(inp: Any) -> Any:
            x = Embedding(i_dim, o_dim, input_length=input_length, trainable=True)(inp)
            x = Dropout(.3)(x)
            return x

        return _nn


###


class DenseLayers():

    @staticmethod
    def regularized() -> Dense:
        return Dense(5)


###


class RnnLayers():

    @staticmethod
    def drqa() -> Callable[[Any], Any]:
        units = 128
        initializer = 'glorot_uniform'

        def _lstm() -> LSTM:
            return LSTM(units, dropout=.3, recurrent_initializer=initializer, return_sequences=True)

        def _nn(inp: Any) -> Any:
            x = Bidirectional(_lstm(), merge_mode="concat")(inp)
            # x = Bidirectional(_lstm(), merge_mode="concat")(x)
            # x = Bidirectional(_lstm(), merge_mode="concat")(x)
            return x

        return _nn


###


class AttentionLayers():

    @staticmethod
    def weighted_sum() -> Callable[[Any], Any]:

        def _nn(inp: Any) -> Any:
            # # --> (batch, n_tokens, embedding_dim)
            # scores = Dense(1, use_bias=False)(inp)  # 1 score foreach embedding input
            # # --> (batch, n_tokens, 1)
            # weights = Softmax(axis=1)(scores)
            # # --> (batch, n_tokens, 1)
            # x_weighted = weights * x  # average
            # # --> (batch, n_tokens, 1)
            # weighted_sum = tf.reduce_sum(x_weighted, axis=1)  # sum
            # # --> (batch, n_tokens)
            # return weighted_sum  #, weights

            # --> (batch, n_tokens, embedding_dim)
            q = inp
            # --> (batch, n_tokens, embedding_dim)
            b = Dense(1, activation="softmax", use_bias=False)(q)
            # --> (batch, n_tokens, 1)
            q_weighted = b * q
            # --> (batch, n_tokens, 1)
            weighted_sum = tf.reduce_sum(q_weighted, axis=1)
            # --> (batch, n_tokens)
            return weighted_sum

        return _nn


###


class Seq2SeqLayers():

    @staticmethod
    def encoder() -> Callable[[Any], Tuple[Any, List[Any]]]:
        units = 128

        def _rnn():
            return LSTM(units, return_state=True)

        def _nn(inp: Any) -> Tuple[Any, List[Any]]:
            x, hidden_state, cell_state = _rnn()(inp)
            return x, [hidden_state, cell_state]

        return _nn

    @staticmethod
    def decoder() -> Callable[[Any, List[Any]], Tuple[Any, List[Any]]]:
        units = 128

        def _rnn():
            return LSTM(units, return_sequences=True, return_state=True)

        def _nn(inp: Any, encoder_states: List[Any]) -> Tuple[Any, List[Any]]:
            x, hidden_state, cell_state = _rnn()(inp, initial_state=encoder_states)
            return x, [hidden_state, cell_state]

        return _nn
