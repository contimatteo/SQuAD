from typing import Tuple

from random import randint

import numpy as np
import utils.configs as Configs

###

N_ROWS = 500

###


def X_train_faker() -> np.ndarray:
    ### [Q] tokens
    q_tokens = np.random.random_sample((N_ROWS, Configs.N_QUESTION_TOKENS))
    ### [P] tokens
    p_tokens = np.random.random_sample((N_ROWS, Configs.N_PASSAGE_TOKENS))
    ### [P] exact match
    p_match = np.zeros((N_ROWS, Configs.N_PASSAGE_TOKENS, Configs.DIM_EXACT_MATCH)).astype(np.float)
    ### [P] POS tags
    p_pos = np.zeros((N_ROWS, Configs.N_PASSAGE_TOKENS, Configs.N_POS_CLASSES)).astype(np.float)
    ### [P] NER annotations
    p_ner = np.random.random_sample((N_ROWS, Configs.N_PASSAGE_TOKENS, Configs.N_NER_CLASSES))
    ### [P] TF
    p_tf = np.zeros((N_ROWS, Configs.N_PASSAGE_TOKENS, Configs.DIM_TOKEN_TF)).astype(np.float)

    return [q_tokens, p_tokens, p_match, p_pos, p_ner, p_tf]


def Y_train_faker() -> np.ndarray:
    shape = (N_ROWS, Configs.N_PASSAGE_TOKENS, 2)

    def _generator(n, k):
        """n numbers with sum k"""
        if n == 1:
            return [k]
        num = 0 if k == 0 else randint(1, k)
        return [num] + _generator(n - 1, k - num)

    def _Y_start_end():
        # Y_start = np.array(_generator(Configs.N_PASSAGE_TOKENS, 100), dtype=np.float)
        # Y_start /= 100
        # Y_end = np.array(_generator(Configs.N_PASSAGE_TOKENS, 100), dtype=np.float)
        # Y_end = Y_end[::-1] / 100
        Y_start = np.zeros((Configs.N_PASSAGE_TOKENS, ))
        Y_end = np.zeros((Configs.N_PASSAGE_TOKENS, ))
        Y_start_index = randint(0, Configs.N_PASSAGE_TOKENS - 2)
        Y_end_index = randint(Y_start_index + 1, Configs.N_PASSAGE_TOKENS - 1)
        Y_start[Y_start_index] = 1
        Y_end[Y_end_index] = 1

        return Y_start, Y_end

    Y = np.zeros(shape)
    for i in range(1):
        Y_start, Y_end = _Y_start_end()
        Y_true = np.array([Y_start, Y_end]).T
        Y[i] = Y_true

    return Y.astype(np.float)


###


def __trunc_X_feature_at_n_token(feature, n):
    return feature[:, 0:n]


def __prepare_Y_true_onehot_encoding(Y: np.ndarray) -> int:
    ### --> (n_examples, n_tokens, 2)
    Y_transposed = np.transpose(Y, axes=[0, 2, 1])
    ### --> (n_examples, 2, n_tokens)

    ### existing at least one probability not equal to zero?
    Y_onehot_label_missing = np.any(Y_transposed, axis=2)  ### --> (n_examples, 2)
    ### invert each boolean value: now we have `1` where we have all zero probabilities.
    Y_onehot_label_missing = np.invert(Y_onehot_label_missing)  ### --> (n_examples, 2)
    ### cast `bool` values to `int`
    Y_onehot_label_missing = Y_onehot_label_missing.astype(int)  ### --> (n_examples, 2)

    ### (n_examples, n_tokens, 2) and (n_examples, 2) --> (n_tokens+1, n_examples, 2)
    Y_onehot_with_additional_case = np.concatenate(
        (np.transpose(Y, axes=[1, 0, 2]), np.expand_dims(Y_onehot_label_missing, axis=0))
    )
    Y_onehot_with_additional_case = np.transpose(Y_onehot_with_additional_case, axes=[1, 0, 2])

    return Y_onehot_with_additional_case


def XY_data_from_dataset(data) -> Tuple[np.ndarray]:

    assert isinstance(data, tuple)
    assert len(data) == 8

    #

    p_tokens = data[1]
    p_tokens = __trunc_X_feature_at_n_token(p_tokens, Configs.N_PASSAGE_TOKENS)
    assert isinstance(p_tokens, np.ndarray)
    assert len(p_tokens.shape) == 2
    assert p_tokens.shape[1] == Configs.N_PASSAGE_TOKENS

    q_tokens = data[2]
    q_tokens = __trunc_X_feature_at_n_token(q_tokens, Configs.N_PASSAGE_TOKENS)
    assert isinstance(q_tokens, np.ndarray)
    assert len(q_tokens.shape) == 2
    assert q_tokens.shape[1] == Configs.N_QUESTION_TOKENS

    labels = data[3]
    labels = __trunc_X_feature_at_n_token(labels, Configs.N_PASSAGE_TOKENS)
    assert isinstance(labels, np.ndarray)
    assert len(labels.shape) == 3
    assert labels.shape[1] == Configs.N_PASSAGE_TOKENS
    assert labels.shape[2] == 2

    p_pos = data[4]
    p_pos = __trunc_X_feature_at_n_token(p_pos, Configs.N_PASSAGE_TOKENS)
    assert isinstance(p_pos, np.ndarray)
    assert len(p_pos.shape) == 3
    assert p_pos.shape[1] == Configs.N_PASSAGE_TOKENS
    assert p_pos.shape[2] == Configs.N_POS_CLASSES

    p_ner = data[5]
    p_ner = __trunc_X_feature_at_n_token(p_ner, Configs.N_PASSAGE_TOKENS)
    assert isinstance(p_ner, np.ndarray)
    assert len(p_ner.shape) == 3
    assert p_ner.shape[1] == Configs.N_PASSAGE_TOKENS
    assert p_ner.shape[2] == Configs.N_NER_CLASSES

    p_tf = data[6]
    p_tf = __trunc_X_feature_at_n_token(p_tf, Configs.N_PASSAGE_TOKENS)
    assert isinstance(p_tf, np.ndarray)
    assert len(p_tf.shape) == 2
    assert p_tf.shape[1] == Configs.N_PASSAGE_TOKENS

    p_match = data[7]
    p_match = __trunc_X_feature_at_n_token(p_match, Configs.N_PASSAGE_TOKENS)
    assert isinstance(p_match, np.ndarray)
    assert len(p_match.shape) == 3
    assert p_match.shape[1] == Configs.N_PASSAGE_TOKENS
    assert p_match.shape[2] == Configs.DIM_EXACT_MATCH

    #

    X = [q_tokens, p_tokens, p_match, p_pos, p_ner, p_tf]

    Y = __prepare_Y_true_onehot_encoding(labels)

    #

    return X, Y
