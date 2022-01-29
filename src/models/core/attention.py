from typing import Callable, Any, List

import tensorflow as tf

from tensorflow.keras.layers import Dot, Dense
from tensorflow.keras.activations import softmax

###


# pylint: disable=invalid-name
def AttentionCore(compatibility: Callable[[Any, Any], Any],
                  distribution: Callable[[Any], Any]) -> Callable[[List[Any]], Any]:

    def _nn(K_q: List[Any]) -> Any:
        keys, query = K_q[0], K_q[1]

        energy_scores = compatibility(keys, query)

        attention_weights = distribution(energy_scores)

        return attention_weights

    return _nn


# pylint: disable=invalid-name
def AttentionModel(compatibility: Callable[[Any, Any], Any],
                   distribution: Callable[[Any], Any]) -> Callable[[List[Any]], Any]:

    def _nn(K_q: List[Any]) -> Any:
        keys, query = K_q[0], K_q[1]

        attention_weights = AttentionCore(compatibility, distribution)([keys, query])

        weighted_values = attention_weights * keys

        return tf.reduce_sum(weighted_values, axis=1)

    return _nn


###


# pylint: disable=invalid-name
def WeightedSumSelfAttention():
    W = Dense(1, use_bias=False)

    def compatibility(keys: Any, *_) -> Any:
        return W(keys)

    def distribution(scores: Any) -> Any:
        return softmax(scores)

    return AttentionModel(compatibility, distribution)


# pylint: disable=invalid-name
def BiLinearSimilarityAttention():
    # W_start = Dense(256, activation="exponential", use_bias=False)
    # W_end = Dense(256, activation="exponential", use_bias=False)
    #
    # ISSUE: the `compatibility` function MUST take as input `K, q` where `K` is a matrix.
    # def compatibility(W: Any) -> Callable[[Any], Callable[[Any, Any], Any]]:
    #     def _W_compatibility(key: Any, queries: Any) -> Any:
    #         scores = []
    #         for idx in range(queries.shape[1]):
    #             query = queries[:, idx, :]
    #             ### --> (_, 256)
    #             score = Dot(axes=1)([key, W(query)])
    #             ### --> (_, 1)
    #             scores.append(score)
    #         scores = tf.convert_to_tensor(scores)
    #         ### --> (40, _, 1)
    #         scores = tf.transpose(scores, perm=[1, 0, 2])
    #         ### --> (_, 40, 1)
    #         scores = tf.squeeze(scores, axis=[2])
    #         ### --> (_, 40)
    #         return scores
    #     return _W_compatibility
    # def distribution(scores: Any) -> Any:
    #     return softmax(scores)
    #
    # start_probability = AttentionCore(compatibility(W_start), distribution)
    # end_probability = AttentionCore(compatibility(W_end), distribution)
    #
    # probabilities = tf.convert_to_tensor([start_probability, end_probability])
    # ### --> (2, _, 40)
    # probabilities = tf.transpose(probabilities, perm=[1, 2, 0])
    # ### --> (_, 40, 2)
    #
    # return probabilities

    pass
