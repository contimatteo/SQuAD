from typing import Callable, Any, List, Tuple

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Dot
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Concatenate

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

        # pylint: disable=unused-argument
        def compatibility_func(query: Any, keys: Any) -> Any:
            # TODO: missing computation steps ...
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

        def _alpha() -> Callable[[Any], Any]:
            return Dense(1, activation="relu")

        def _compatibility_func(a: Any, b: Any) -> Any:
            ### it's actually scalar product
            return a * b

        def _distribution_func() -> Callable[[Any], Any]:
            return Softmax()

        def _custom_core(query: Any, token_index: Any, alpha_keys: Any) -> Any:
            token_query = query[:, token_index, :]
            # (batch_size,token_length)
            token_query = tf.expand_dims(token_query, axis=1)
            # (batch_size,1,token_length)

            alpha_token_query = _alpha()(token_query)
            # (batch_size,1,1)

            energy_scores = _compatibility_func(alpha_keys, alpha_token_query)
            # (batch_size,keys_length,1)
            attention_weights = _distribution_func()(energy_scores)
            # (batch_size,keys_length,1)

            return attention_weights

        def _nn(passage_and_question: List[Any]) -> Any:
            passage, question = passage_and_question[0], passage_and_question[1]

            alpha_question = _alpha()(question)
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


###


class Customlayers():

    @staticmethod
    def embeddings_similarity() -> Callable[[Any, Any], Any]:
        ###
        ### In the prof. slides this layer is called "Bilinear Attention".
        ### TODO: try to investigate this definition ...
        ###
        ### There are one huge similarity between the below steps and the Attention mechanism:
        ###   - the `cosine_similarity` is computed starting from 2 vectors and it's the Dot
        ###     product between these ones. At the same level we have the `compatibility
        ###     function` inside Attention computed between the `query` and the `keys` vectors.
        ###
        ### The only thing that is missing (in the steps below) is somenthing similar to the
        ### `weighted_average` step of Attention. This lack could be justified by the fact that
        ### inside the Attention layer we compute the relevance of each `key` vector respect
        ### to the `query` vector and, as a consequence, at the end we have to average the relevance
        ### (weighted vector of scores produced by the `compatibility function`) of the `query`
        ### compared to all the `key` vectors in order to produce a single vector.
        ###
        ### Conversely, in this layer we have to compute the `attention/similarity` for each of the
        ### output vectors of the Passage RNN.
        ###
        ### Summary of the similarities between this steps and the Attention mechanism:
        ###  • `query` --> `q_encoding`
        ###  • `keys` --> `p_tokens`
        ###  • `compatibility_fun()` --> `_nn_similarity_classifier()`
        ###  • `distribution_fun()` --> ?????
        ###  • `weighted_sum()` --> ?????
        ###

        def _cosine_similarity(vect1, vect2):
            return Dot(axes=1, normalize=True)([vect1, vect2])

        def _similarity_features(q, p):
            ### TODO: consider one possible improvement by passing to the similarity classifier not
            ### only the similarity vector but also the original `q` and `p` vectors.
            # features = Concatenate()[q, p, _cosine_similarity(q, p)]
            features = _cosine_similarity(q, p)
            return features

        def _nn_similarity(q, p):
            x = _similarity_features(q, p)

            # x = Dense(128, activation="relu")(x)
            x = Dense(32, activation="sigmoid")(x)
            x = Dense(4, activation="sigmoid")(x)

            return x

        def _nn(passage_and_question: List[Any]) -> Any:
            _similarity_scores = []
            p_tokens, q_encoding = passage_and_question[0], passage_and_question[1]

            for p_token_idx in range(p_tokens.shape[1]):
                # --> (None, 1, 256)
                p_token = p_tokens[:, p_token_idx, :]
                # --> (None, 256)
                similarity = _nn_similarity(q_encoding, p_token)
                # --> (None, 4)
                similarity = tf.expand_dims(similarity, axis=[1])
                # --> (None, 1, 4)
                _similarity_scores.append(similarity)

            # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            return tf.concat(_similarity_scores, axis=1)

        return _nn
