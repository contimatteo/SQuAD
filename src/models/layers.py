from typing import Callable, Any, List, Tuple

import tensorflow as tf

from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Lambda
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
    '''
    ### Input:
     - @param `query`
     - @param `keys`

    ### Computation
     1. `energy_scores = compatibility_func(query, keys)`
     2. `attention_weights = distribution_func(energy_scores)`
     3. `weighted_values = values ? attention_weights * values : attention_weights * keys`
     4. `context_vector = tf.reduce_sum(weighted_values)`

    ### Output
     - @param `context_vector`
     - @param `attention_weights`
    '''

    # @staticmethod
    # def core(compatibility_func, distribution_func):
    #     def _nn(query, keys):
    #         energy_scores = compatibility_func(query, keys)
    #         attention_weights = distribution_func(energy_scores)
    #         weighted_values = attention_weights * keys
    #         context_vector = tf.reduce_sum(weighted_values)
    #         return context_vector
    #     return _nn

    # @staticmethod
    # def weighted_sum() -> Callable[[Any], Any]:
    #     def _nn(inp: Any) -> Any:
    #         # --> (batch, n_tokens, embedding_dim)
    #         q = inp
    #         # --> (batch, n_tokens, embedding_dim)
    #         b = Dense(1, activation="softmax", use_bias=False)(q)
    #         # --> (batch, n_tokens, 1)
    #         q_weighted = b * q
    #         # --> (batch, n_tokens, 1)
    #         weighted_sum = tf.reduce_sum(q_weighted, axis=1)
    #         # --> (batch, n_tokens)
    #         return weighted_sum
    #     return _nn

    @staticmethod
    def weighted_sum(axis: int) -> Callable[[Any, Any], Any]:

        def _weight_and_sum(weights_and_values: List[Any]) -> Any:
            weights, values_to_sum = weights_and_values[0], weights_and_values[1]
            values_weighted = weights * values_to_sum
            return tf.reduce_sum(values_weighted, axis=axis)

        return _weight_and_sum
        # return Lambda(_weight_and_sum)

    #

    @staticmethod
    def question_encoding() -> Callable[[Any], Any]:

        def _nn(keys: Any) -> Any:
            compatibility_func = Dense(1, use_bias=False)
            distribution_func = Softmax()

            # --> (batch, n_tokens, embedding_dim)
            energy_scores = compatibility_func(keys)
            # --> (batch, n_tokens, 1)
            attention_weights = distribution_func(energy_scores)
            # --> (batch, n_tokens, 1)
            weighted_sum = AttentionLayers.weighted_sum(axis=1)([attention_weights, keys])
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
