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
    W = Dense(1, use_bias=False)

    def compatibility(keys: Any, *_) -> Any:
        return W(keys)

    def distribution(scores: Any) -> Any:
        return softmax(scores)

    return AttentionModel(compatibility, distribution)


# pylint: disable=invalid-name
def BiLinearSimilarityAttention():
    Ws = Dense(256, activation="exponential", use_bias=False)
    We = Dense(256, activation="exponential", use_bias=False)

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


# pylint: disable=invalid-name
def AlignedAttention():
    alpha = Dense(1, use_bias=False)

    def compatibility(keys: Any, query: Any) -> Any:
        _alpha_query = alpha(query)

        scores = []
        for idx in keys.shape[1]:
            key = key[:, idx, :]
            _alpha_key = alpha(key)
            scores.append(_alpha_key * _alpha_query)

        return tf.convert_to_tensor(scores)

    def distribution(scores: Any) -> Any:
        return softmax(scores)

    def _nn(keys_and_queries: List[Any]):
        keys, queries = keys_and_queries[0], keys_and_queries[1]

        weights = []
        for idx in queries.shape[1]:
            query = queries[:, idx, :]
            weights.append(AttentionModel(compatibility, distribution)([keys, query]))

        return tf.convert_to_tensor(weights)

    return _nn
