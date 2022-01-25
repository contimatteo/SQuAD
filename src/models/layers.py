from typing import Callable, Any, List, Tuple

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Lambda, Multiply
from tensorflow.keras.layers import Reshape
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
    ## Unified Model Components

    #### INPUT:
     - @param `query`
     - @param `keys`
     - @param `values` (optional)
   
    #### COMPUTATION
     1. `energy_scores = compatibility_func(query, keys)`
     2. `attention_weights = distribution_func(energy_scores)`
     3. `weighted_values = values ? attention_weights * values : attention_weights * keys`
     4. `context_vector = tf.reduce_sum(weighted_values)`

    #### OUTPUT
     - @param `context_vector`
     - @param `attention_weights` (optional)
    '''

    @staticmethod
    def weighted_sum(axis: int) -> Callable[[Any, Any], Any]:

        def _weight_and_sum(weights_and_values: List[Any]) -> Any:
            weights, values_to_sum = weights_and_values[0], weights_and_values[1]
            values_weighted = weights * values_to_sum
            return tf.reduce_sum(values_weighted, axis=axis)

        return _weight_and_sum

    @staticmethod
    def core(compatibility_func, distribution_func):

        def _nn(keys, query=None):
            energy_scores = compatibility_func(keys, query)
            attention_weights = distribution_func(energy_scores)
            context_vector = AttentionLayers.weighted_sum(axis=1)([attention_weights, keys])
            return context_vector

        return _nn

    #

    @staticmethod
    def question_encoding() -> Callable[[Any], Any]:

        def compatibility_func(keys, *_):
            return Dense(1, use_bias=False)(keys)

        def distribution_func(scores):
            return Softmax()(scores)

        def _nn(keys: Any) -> Any:
            # energy_scores = compatibility_func(keys)
            # attention_weights = distribution_func(energy_scores)
            # weighted_sum = AttentionLayers.weighted_sum(axis=1)([attention_weights, keys])
            # return weighted_sum
            return AttentionLayers.core(compatibility_func, distribution_func)(keys, None)

        return _nn

    @staticmethod
    def passage_embeddings() -> Callable[[Any, Any], Any]:

        def compatibility_func(query: Any, keys: Any) -> Any:
            ### TODO: missing computation steps ...
            # q_scores = Dense(1, activation="relu", use_bias=False)(query)
            # k_scores = Dense(1, activation="relu", use_bias=False)(keys)
            # return tf.matmul(q_scores, k_scores, transpose_b=True)
            return Dense(1, activation="relu", use_bias=False)(query)

        def distribution_func(scores):
            return Softmax()(scores)

        def _nn(query_and_keys: List[Any]) -> Any:
            query, keys = query_and_keys[0], query_and_keys[1]
            energy_scores = compatibility_func(query, keys)
            attention_weights = distribution_func(energy_scores)
            weighted_sum = AttentionLayers.weighted_sum(axis=1)([attention_weights, keys])
            return weighted_sum

        return _nn

    #

    @staticmethod
    def alignment() -> Callable[[Any, Any], Any]:

        def alpha():
            return Dense(1, activation="relu")

        def _compatibility_func(a: Any, b: Any) -> Any:
            #it's actually scalar product
            return a * b

        def _distribution_func():
            return Softmax()

        def _custom_core(query: Any, token_index: Any, alpha_keys: Any) -> Any:
            token_query = query[:, token_index, :]
            # (batch_size,token_length)
            token_query = tf.expand_dims(token_query, axis=1)
            # (batch_size,1,token_length)

            alpha_token_query = alpha()(token_query)
            # (batch_size,1,1)

            energy_scores = _compatibility_func(alpha_keys, alpha_token_query)
            # (batch_size,keys_length,1)
            attention_weights = _distribution_func()(energy_scores)
            # (batch_size,keys_length,1)

            return attention_weights

        def _nn(passage: Any, question: Any):
            alpha_question = alpha()(question)

            aligned_tokens = []
            for i in range(passage.shape[1]):

                attention_weights = _custom_core(passage, i, alpha_question)
                # (batch_size,question length,1)

                context_vector = attention_weights * question
                # (batch_size,question length,token_length)
                context_vector = tf.reduce_sum(context_vector, axis=1)
                # (batch_size,token_length)

                context_vector = tf.expand_dims(context_vector, axis=1)
                # (batch_size,1,token_length)

                aligned_tokens.append(context_vector)

            aligned_passage = tf.concat(aligned_tokens, axis=1)
            # (batch_size,passage_length,token_length)

            return aligned_passage

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
