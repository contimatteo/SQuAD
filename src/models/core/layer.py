from typing import Callable, Any, List

import tensorflow as tf
import numpy as np

import utils.configs as Configs

from tensorflow.keras.layers import Dense, Dropout, Dot
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.keras.activations import softmax
from tensorflow.keras.initializers import Constant

###


class EmbeddingLayers():

    @staticmethod
    def glove(input_length: int, glove_matrix: np.ndarray) -> Callable[[Any], Any]:
        assert isinstance(input_length, int)

        i_dim = glove_matrix.shape[0]  # size of the vocabulary
        o_dim = Configs.DIM_EMBEDDING  # dimension of the 'dense' embedding

        def _nn(inp: Any) -> Any:
            x = Embedding(
                i_dim,
                o_dim,
                input_length=input_length,
                embeddings_initializer=Constant(glove_matrix),
                trainable=False
            )(inp)
            x = Dropout(.3)(x)
            return x

        return _nn


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
        W = Dense(1, use_bias=False)

        def compatibility_func(keys, *_):
            return W(keys)

        def distribution_func(scores):
            return softmax(scores)

        def _nn(keys: Any) -> Any:
            return AttentionLayers.core(compatibility_func, distribution_func)(keys, None)

        return _nn

    @staticmethod
    def alignment() -> Callable[[Any, Any], Any]:
        ### TODO: exploit the `AttentionLayers.core()` function instead of
        ### replicating all the common steps of Attention core mechanism.

        _alpha = Dense(1, activation="relu")

        def compatibility(a: Any, b: Any) -> Any:
            return a * b

        def distribution(scores: Any) -> Callable[[Any], Any]:
            return softmax(scores)

        def _custom_core(query: Any, token_index: Any, alpha_keys: Any) -> Any:
            token_query = query[:, token_index, :]
            # (batch_size, token_length)
            token_query = tf.expand_dims(token_query, axis=1)
            # (batch_size, 1, token_length)

            alpha_token_query = _alpha(token_query)
            # (batch_size, 1, 1)

            energy_scores = compatibility(alpha_keys, alpha_token_query)
            # (batch_size, keys_length,1)
            attention_weights = distribution(energy_scores)
            # (batch_size, keys_length,1)

            return attention_weights

        def _nn(passage_and_question: List[Any]) -> Any:
            passage, question = passage_and_question[0], passage_and_question[1]

            alpha_question = _alpha(question)
            aligned_tokens = []

            for i in range(passage.shape[1]):
                attention_weights = _custom_core(passage, i, alpha_question)
                ### (batch_size, question_length, 1)

                context_vector = attention_weights * question
                ### (batch_size, question_length,token_length)
                context_vector = tf.reduce_sum(context_vector, axis=1)
                ### (batch_size, token_length)

                context_vector = tf.expand_dims(context_vector, axis=1)
                ### (batch_size, 1, token_length)

                aligned_tokens.append(context_vector)

            # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            aligned_passage = tf.concat(aligned_tokens, axis=1)
            ### (batch_size, passage_length, token_length)

            return aligned_passage

        return _nn

    #

    @staticmethod
    def bilinear_similarity() -> Callable[[Any, Any], Any]:
        Ws = Dense(256, activation="exponential", use_bias=False)
        We = Dense(256, activation="exponential", use_bias=False)

        def compatibility(w_type: str) -> Callable[[Any, Any], Any]:
            W = Ws if w_type == "start" else We

            def _scores(keys, query) -> Any:
                scores = []
                for key_idx in range(keys.shape[1]):
                    k = keys[:, key_idx, :]  ### --> (_, 256)
                    q = W(query)
                    score = Dot(axes=1, normalize=True)([k, q])  ### --> (_, 1)
                    scores.append(score)

                ### --> (n_tokens, _, 1)
                scores = tf.convert_to_tensor(scores)
                ### --> (n_tokens, _, 1)
                scores = tf.transpose(scores, perm=[1, 0, 2])
                ### --> (_, n_tokens, 1)
                scores = tf.squeeze(scores, axis=[2])
                ### --> (_, n_tokens)

                return scores

            return _scores

        def distribution(scores):
            return softmax(scores)

        def _nn(keys_and_queries: List[Any]) -> Any:
            keys = keys_and_queries[0]  ### --> (_, n_tokens, 256)
            query = keys_and_queries[1]  ### --> (_, 256)

            s_scores = compatibility("start")(keys, query)
            s_weights = distribution(s_scores)

            e_scores = compatibility("end")(keys, query)
            e_weights = distribution(e_scores)

            probs = tf.convert_to_tensor([s_weights, e_weights])
            ### --> (2, _, 40)
            probs = tf.transpose(probs, perm=[1, 2, 0])
            ### --> (_, 40, 2)

            return probs

        return _nn
