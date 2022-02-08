# pylint: disable=unused-import
from typing import Any, Tuple, List, Dict

import os
import json
import numpy as np
import tensorflow as tf

from alive_progress import alive_bar
from tensorflow.keras.backend import set_learning_phase

import utils.env_setup
import utils.configs as Configs

from data import get_data, load_data
from models import DRQA
from utils import LocalStorageManager
from utils import X_data_from_dataset, Y_data_from_dataset, QP_data_from_dataset

###

os.environ["WANDB_JOB_TYPE"] = "training"

set_learning_phase(0)

LocalStorage = LocalStorageManager()

N_ROWS_SUBSET = None  # `None` for all rows :)

###


def __predict():
    X, _ = X_data_from_dataset(get_data("features"), N_ROWS_SUBSET)
    glove_matrix = get_data("glove")

    model = DRQA(glove_matrix)

    nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    assert nn_checkpoint_directory.is_file()

    ### load weights
    model.load_weights(str(nn_checkpoint_directory))

    Y_pred = model.predict(X, batch_size=Configs.NN_BATCH_SIZE, verbose=1)

    return Y_pred


def __compute_answers_tokens_indexes(Y: np.ndarray) -> Dict[str, np.ndarray]:
    _, question_indexes = X_data_from_dataset(get_data("features"), N_ROWS_SUBSET)
    question_indexes_unique = list(np.unique(question_indexes))

    answers_tokens_probs_map = {}

    with alive_bar(len(question_indexes_unique)) as progress_bar:
        for question_index in question_indexes_unique:
            subset_indexes = np.array(question_indexes == question_index)

            current_answers = Y[subset_indexes]
            answers_tokens_probs_map[question_index] = current_answers

            progress_bar()

    assert len(answers_tokens_probs_map.keys()) == len(question_indexes_unique)

    #

    __weigth_answer_probs = lambda answer: answer[0:-1] - answer[-1]

    answers_tokens_indexes_map: Dict[str, np.ndarray] = {}

    with alive_bar(len(list(answers_tokens_probs_map.items()))) as progress_bar:
        for (q_index, answers) in answers_tokens_probs_map.items():
            answer_tokens_probs = np.array(
                [__weigth_answer_probs(answers[idx]) for idx in range(answers.shape[0])],
                dtype=answers.dtype
            )

            start_index = np.argmax(answer_tokens_probs[:, :, 0].flatten())
            end_index = np.argmax(answer_tokens_probs[:, :, 1].flatten())

            answers_tokens_indexes_map[q_index] = np.array([start_index, end_index], dtype=np.int16)

            progress_bar()

    #

    return answers_tokens_indexes_map


def __compute_answers_predictions(answers_tokens_indexes_map: Any) -> Dict[str, str]:
    answers_for_question_map = {}

    qids, questions, passages = QP_data_from_dataset(get_data("original"))

    with alive_bar(qids.shape[0]) as progress_bar:
        for (idx, qid) in enumerate(list(qids)):
            answer = ""
            passage_tokens = passages[idx]
            # passage = " ".join(passage_tokens)

            if qid in answers_tokens_indexes_map:
                answ_tokens_bounds = answers_tokens_indexes_map[qid]
                answ_token_start_index = answ_tokens_bounds[0]
                answer_token_end_index = answ_tokens_bounds[1]

                ### INFO: the original predictions are based on PADDED passages.
                if answer_token_end_index >= len(passage_tokens):
                    answer_token_end_index = len(passage_tokens) - 1
                if answ_token_start_index >= len(passage_tokens):
                    answ_token_start_index = len(passage_tokens) - 1

                ### INFO: we have no guarantees to have a 'valid' indexes range.
                if answer_token_end_index < answ_token_start_index:
                    answer_token_end_index = answ_token_start_index

                # answ_span_pre_start = passage_tokens[0:answ_token_start_index]
                # answ_span_to_end = passage_tokens[0:answer_token_end_index + 1]
                # ### INFO: we have always to consider the ENTIRE end token.
                # answ_span_to_end += [str(passage_tokens[answer_token_end_index])]
                # ### compute the chars range indexes
                # answ_char_start_index = len(" ".join(answ_span_pre_start))
                # answ_char_end_index = len(" ".join(answ_span_to_end))
                # ### extract answer from chars indexes
                # answer = passage[answ_char_start_index:answ_char_end_index]
                answer = " ".join(passage_tokens[answ_token_start_index:answer_token_end_index + 1])
                answer = str(answer).strip()

            if qid in answers_tokens_indexes_map:
                answers_for_question_map[qid] = answer
            # answers_for_question_map[qid] = answer

            progress_bar()

    return answers_for_question_map


def __store_answers_predictions(answers_predictions_map: Dict[str, str], file_name: str) -> None:
    assert isinstance(answers_predictions_map, dict)

    json_file_url = LocalStorage.answers_predictions_url(file_name)

    if json_file_url.exists():
        json_file_url.unlink()

    with open(str(json_file_url), "w", encoding='utf-8') as file:
        json.dump(answers_predictions_map, file)


###


def test():
    Y_pred = __predict()

    answers_tokens_indexes = __compute_answers_tokens_indexes(Y_pred)
    answers_for_question = __compute_answers_predictions(answers_tokens_indexes)
    __store_answers_predictions(answers_for_question, "training.pred")

    print()
    print("The generated answers (json with predictions) file is available at:")
    print(str(LocalStorage.answers_predictions_url("training.pred")))
    print()

    #

    # ### TODO: remove the following code ...
    # Y_true = Y_data_from_dataset(get_data("labels"), N_ROWS_SUBSET)

    # ### TODO: remove the following code ...
    # answers_tokens_indexes = __compute_answers_tokens_indexes(Y_true)
    # answers_for_question = __compute_answers_predictions(answers_tokens_indexes)
    # __store_answers_predictions(answers_for_question, "training.true")


###

if __name__ == "__main__":
    load_data()

    test()
