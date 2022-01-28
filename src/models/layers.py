from typing import Callable, Any, List

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Dot
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.keras.activations import softmax

###


class EmbeddingLayers():

    @staticmethod
    def glove(input_length: int) -> Callable[[Any], Any]:
        assert isinstance(input_length, int)

        i_dim = 100  # size of the vocabulary
        o_dim = 50  # dimension of the dense embedding

        ### TODO: change params:
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
            ### TODO: use `Dot` layer ...
            return a * b

        def distribution(scores: Any) -> Callable[[Any], Any]:
            return softmax(scores)

        def _custom_core(query: Any, token_index: Any, alpha_keys: Any) -> Any:
            token_query = query[:, token_index, :]
            # (batch_size,token_length)
            token_query = tf.expand_dims(token_query, axis=1)
            # (batch_size,1,token_length)

            alpha_token_query = _alpha(token_query)
            # (batch_size,1,1)

            energy_scores = compatibility(alpha_keys, alpha_token_query)
            # (batch_size,keys_length,1)
            attention_weights = distribution(energy_scores)
            # (batch_size,keys_length,1)

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

        def compatibility_start(key, query):
            return Dot(axes=1)([key, Ws(query)])

        def compatibility_end(key, query):
            return Dot(axes=1)([key, We(query)])

        def distribution(scores):
            return softmax(scores)

        def _nn(passage_and_question: List[Any]) -> Any:
            queries = passage_and_question[0]  ### --> (_, n_tokens, 256)
            key = passage_and_question[1]  ### --> (_, 256)

            ### START

            start_scores = []
            for query_idx in range(queries.shape[1]):
                query = queries[:, query_idx, :]
                start_score = compatibility_start(key, query)
                start_scores.append(start_score)

            start_scores = tf.convert_to_tensor(start_scores)
            ### --> (40, _, 1)
            start_scores = tf.transpose(start_scores, perm=[1, 0, 2])
            ### --> (_, 40, 1)
            start_scores = tf.squeeze(start_scores, axis=[2])
            ### --> (_, 40)

            start_probability = distribution(start_scores)

            ### START

            end_scores = []

            for query_idx in range(queries.shape[1]):
                query = queries[:, query_idx, :]
                score = compatibility_end(key, query)
                end_scores.append(score)

            end_scores = tf.convert_to_tensor(end_scores)
            ### --> (40, _, 1)
            end_scores = tf.transpose(end_scores, perm=[1, 0, 2])
            ### --> (_, 40, 1)
            end_scores = tf.squeeze(end_scores, axis=[2])
            ### --> (_, 40)

            end_probability = distribution(end_scores)

            ###

            probabilities = tf.convert_to_tensor([start_probability, end_probability])
            ### --> (2, _, 40)
            probabilities = tf.transpose(probabilities, perm=[1, 2, 0])
            ### --> (_, 40, 2)

            return probabilities

        return _nn
