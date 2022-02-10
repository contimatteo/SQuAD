from typing import Callable, Any, List

import tensorflow as tf

from tensorflow.keras.layers import Dot, Dense
from tensorflow.keras.activations import softmax

###


# pylint: disable=invalid-name
def AttentionCore(compatibility: Callable[[Any, Any], Any],
                  distribution: Callable[[Any], Any]) -> Callable[[List[Any]], Any]:

    def _nn(keys_and_query: List[Any]) -> Any:
        keys, query = keys_and_query[0], keys_and_query[1]

        energy_scores = compatibility(keys, query)

        attention_weights = distribution(energy_scores)

        return attention_weights

    return _nn


# pylint: disable=invalid-name
def AttentionModel(compatibility: Callable[[Any, Any], Any],
                   distribution: Callable[[Any], Any]) -> Callable[[List[Any]], Any]:

    def _nn(keys_and_query: List[Any]) -> Any:
        keys, query = keys_and_query[0], keys_and_query[1]

        attention_weights = AttentionCore(compatibility, distribution)([keys, query])

        weighted_values = attention_weights * keys

        return tf.reduce_sum(weighted_values, axis=1)

    return _nn


###


# pylint: disable=invalid-name
def WeightedSumSelfAttention():
    W = Dense(
        1,
        use_bias=False,
        kernel_regularizer='l2',
        activity_regularizer='l2',
        bias_regularizer='l2'
    )

    def compatibility(keys: Any, *_) -> Any:
        return W(keys)

    def distribution(scores: Any) -> Any:
        return softmax(scores)

    def _nn(keys: Any):
        return AttentionModel(compatibility, distribution)([keys, None])

    return _nn


###


# pylint: disable=invalid-name
def AlignedAttention() -> Callable[[Any, Any], Any]:
    ### TODO: exploit the `AttentionLayers.core()` function instead of
    ### replicating all the common steps of Attention core mechanism.

    _alpha = Dense(
        1,
        activation="relu",
        kernel_regularizer='l2',
        activity_regularizer='l2',
        bias_regularizer='l2'
    )

    def compatibility(a: Any, b: Any) -> Any:
        alpha_a = _alpha(a)
        # (batch_size,question_length,units)
        alpha_b = _alpha(b)
        # (batch_size,1,units)

        shape = alpha_b.shape
        shape = [-1, shape[2], shape[1]]
        alpha_b = tf.reshape(alpha_b, shape)
        # (batch_size,units,1)

        # dot product
        return alpha_a @ alpha_b

    def distribution(scores: Any) -> Callable[[Any], Any]:
        return softmax(scores)

    def _nn(passage_and_question: List[Any]) -> Any:
        passage, question = passage_and_question[0], passage_and_question[1]

        aligned_tokens = []
        for i in range(passage.shape[1]):
            token_passage = passage[:, i, :]  # raw Query
            # (batch_size,token_length)
            token_passage = tf.expand_dims(token_passage, axis=1)
            # (batch_size,1,token_length)

            context_vector = AttentionModel(compatibility, distribution)([question, token_passage])
            # (batch_size, token_length)
            context_vector = tf.expand_dims(context_vector, axis=1)
            # (batch_size, 1, token_length)
            aligned_tokens.append(context_vector)

        # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
        aligned_passage = tf.concat(aligned_tokens, axis=1)
        ### (batch_size, passage_length, token_length)

        return aligned_passage

    return _nn


###


# pylint: disable=invalid-name
def BiLinearSimilarityAttention():
    Ws = Dense(
        256,
        activation="exponential",
        use_bias=False,
        kernel_regularizer='l2',
        activity_regularizer='l2',
        bias_regularizer='l2'
    )
    We = Dense(
        256,
        activation="exponential",
        use_bias=False,
        kernel_regularizer='l2',
        activity_regularizer='l2',
        bias_regularizer='l2'
    )

    def compatibility(w_type: str) -> Callable[[Any, Any], Any]:
        W = Ws if w_type == "start" else We

        def _similarity(a: Any, b: Any) -> Any:
            return Dot(axes=1, normalize=True)([a, b])

        def _scores(keys, query) -> Any:
            scores = []
            for key_idx in range(keys.shape[1]):
                k = keys[:, key_idx, :]  ### --> (_, 256)
                q = W(query)  ### --> (_, 256)
                score = _similarity(k, q)  ### --> (_, 1)
                scores.append(score)

            scores = tf.convert_to_tensor(scores)  ### --> (n_tokens, _, 1)
            scores = tf.transpose(scores, perm=[1, 0, 2])  ### --> (_, n_tokens, 1)
            scores = tf.squeeze(scores, axis=[2])  ### --> (_, n_tokens)

            return scores

        return _scores

    def distribution(scores):
        return softmax(scores)

    def _nn(keys_and_queries: List[Any]) -> Any:
        s_weights = AttentionCore(compatibility("start"), distribution)(keys_and_queries)
        e_weights = AttentionCore(compatibility("end"), distribution)(keys_and_queries)

        probs = tf.convert_to_tensor([s_weights, e_weights])  ### --> (2, _, n_tokens)
        probs = tf.transpose(probs, perm=[1, 2, 0])  ### --> (_, n_tokens, 2)

        return probs

    return _nn
