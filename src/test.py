# pylint: disable=unused-import
from typing import Any, Tuple, List, Dict

import os
import json
import numpy as np

import utils.env_setup
import utils.configs as Configs

from data import get_data
from models import DRQA
from utils import XY_data_from_dataset, LocalStorageManager, QP_data_from_dataset

###

os.environ["WANDB_JOB_TYPE"] = "training"

LocalStorage = LocalStorageManager()

###

_, dataset, glove_matrix, _ = get_data(300)


def __dataset() -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray]:
    x, y, question_indexes = XY_data_from_dataset(
        dataset, Configs.NN_BATCH_SIZE * Configs.N_KFOLD_BUCKETS
    )

    return x, y, question_indexes


def __predict(X):
    model = DRQA(glove_matrix)

    nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    assert nn_checkpoint_directory.is_file()

    ### load weights
    model.load_weights(str(nn_checkpoint_directory))

    Y_pred = model.predict(X)

    return Y_pred


def __compute_answers_tokens_indexes(Y: np.ndarray,
                                     question_indexes: np.ndarray) -> Dict[str, np.ndarray]:
    answers_tokens_probs_map = {}

    for question_index in np.unique(question_indexes):
        subset_indexes = np.array(question_indexes == question_index)

        current_answers = Y[subset_indexes]
        answers_tokens_probs_map[question_index] = current_answers

    assert len(answers_tokens_probs_map.keys()) == np.unique(question_indexes).shape[0]

    #

    def __weigth_answer_probs(answer: np.ndarray):
        answer_tokens_probs = answer[0:-1]  ### --> (n_tokens, 2)
        answer_no_token_prob = answer[-1]  ### --> (2,)
        return answer_tokens_probs - answer_no_token_prob

    answers_tokens_indexes_map: Dict[str, np.ndarray] = {}

    for (q_index, answers) in answers_tokens_probs_map.items():
        answer_tokens_probs = np.array(
            [__weigth_answer_probs(answers[idx]) for idx in range(answers.shape[0])],
            dtype=answers.dtype
        )

        start_index = np.argmax(answer_tokens_probs[:, :, 0].flatten())
        end_index = np.argmax(answer_tokens_probs[:, :, 1].flatten())

        answers_tokens_indexes_map[q_index] = np.array([start_index, end_index], dtype=np.int16)

    #

    return answers_tokens_indexes_map


def __compute_answers_predictions(answers_tokens_indexes_map: Any) -> Dict[str, str]:
    answers_for_question_map = {}
    qids, passages = QP_data_from_dataset(dataset)

    for (idx, qid) in enumerate(list(qids)):
        passage = passages[idx]

        if qid in answers_tokens_indexes_map:
            answers_tokens_indexes = answers_tokens_indexes_map[qid]
            answer_start = answers_tokens_indexes[0]
            answer_end = answers_tokens_indexes[1]
            answer = "".join(passage[answer_start:answer_end + 1])
        else:
            answer = ""

        answers_for_question_map[qid] = answer

    return answers_for_question_map


def __store_answers_predictions(answers_predictions_map: Dict[str, str]) -> None:
    assert isinstance(answers_predictions_map, dict)

    json_file_url = LocalStorage.answers_predictions_url("test")

    if json_file_url.exists():
        json_file_url.unlink()

    with open(str(json_file_url), "w", encoding='utf-8') as file:
        json.dump(answers_predictions_map, file)


###


def test():
    X, _, question_indexes = __dataset()

    Y_pred = __predict(X)

    #

    answers_tokens_indexes = __compute_answers_tokens_indexes(Y_pred, question_indexes)

    answers_for_question = __compute_answers_predictions(answers_tokens_indexes)

    __store_answers_predictions(answers_for_question)


###

if __name__ == "__main__":
    test()
