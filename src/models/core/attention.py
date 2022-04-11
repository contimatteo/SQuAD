from typing import Callable, Any, List

import tensorflow as tf

from tensorflow.keras.layers import Dot, Dense
from tensorflow.keras.activations import softmax, exponential

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


class DrQaAttentionLayers:

    @staticmethod
    def PassageAndQuestionAlignment() -> Callable[[Any, Any], Any]:
        ### TODO: exploit the `AttentionLayers.core()` function instead of
        ### replicating all the common steps of Attention core mechanism.

        _alpha = Dense(1, activation="relu")

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
            return softmax(scores, axis=1)
            # return softmax(scores)

        def _nn(passage_and_question: List[Any]) -> Any:
            passage, question = passage_and_question[0], passage_and_question[1]

            aligned_tokens = []
            for i in range(passage.shape[1]):
                token_passage = passage[:, i, :]  # raw Query
                # (batch_size,token_length)
                token_passage = tf.expand_dims(token_passage, axis=1)
                # (batch_size,1,token_length)

                context_vector = AttentionModel(compatibility,
                                                distribution)([question, token_passage])
                # (batch_size, token_length)
                context_vector = tf.expand_dims(context_vector, axis=1)
                # (batch_size, 1, token_length)
                aligned_tokens.append(context_vector)

            # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
            aligned_passage = tf.concat(aligned_tokens, axis=1)
            ### (batch_size, passage_length, token_length)

            return aligned_passage

        return _nn

    @staticmethod
    def BilinearSimilarity():
        Ws = Dense(256, use_bias=False)
        We = Dense(256, use_bias=False)

        dot = Dot(axes=2, normalize=True)

        def __bilinearterm(p_rnn: Any, q_rnn: Any, w_type: str, p_mask):
            # p_rnn:  (None, 50, 256)
            # q_rnn:  (None, 1, 256)

            W = Ws if w_type == "start" else We
            out_W = W(q_rnn)  ### --> (_, 256)

            exp_input = dot([p_rnn, out_W])  ### --> (_, 50, 1)

            exp_out = exponential(exp_input)  ### --> (_, 50, 1)

            return exp_out

        def _nn(passage_and_question: List[Any]) -> Any:
            p_rnn, p_mask, q_rnn = passage_and_question[0], passage_and_question[
                1], passage_and_question[2]
            out = []

            out_start = __bilinearterm(p_rnn, q_rnn, "start", p_mask)
            out.append(out_start)
            out_end = __bilinearterm(p_rnn, q_rnn, "end", p_mask)
            out.append(out_end)

            out = tf.concat(out, axis=2)

            return out

        return _nn
