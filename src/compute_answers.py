# pylint: disable=unused-import,not-callable,import-error
from typing import Any, Dict

import os
import json
import numpy as np

from alive_progress import alive_bar
from tensorflow.keras.backend import set_learning_phase

import utils.env_setup

from data import Dataset
from models import DRQA
from utils import LocalStorageManager, FeaturesUtils, DataUtils
from data import DataReader

###

os.environ["WANDB_JOB_TYPE"] = "training"

set_learning_phase(0)

LocalStorage = LocalStorageManager()

###


def __predict():
    X, _ = FeaturesUtils.X_from_raw_features(Dataset.extract("features"))
    glove_matrix = Dataset.extract("glove")

    model = DRQA(glove_matrix)

    ### load weights
    nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    assert nn_checkpoint_directory.is_file()
    model.load_weights(str(nn_checkpoint_directory))

    Y_pred = model.predict(X, batch_size=128, verbose=1)

    return Y_pred


def __compute_answers_tokens_indexes(Y: np.ndarray) -> Dict[str, np.ndarray]:
    _, question_indexes = FeaturesUtils.X_from_raw_features(Dataset.extract("features"))
    question_indexes_unique = list(np.unique(question_indexes))

    answers_tokens_probs_map = {}

    with alive_bar(len(question_indexes_unique)) as progress_bar:
        for qid in question_indexes_unique:
            subset_indexes = np.array(question_indexes == qid)
            current_answers = Y[subset_indexes]
            answers_tokens_probs_map[qid] = current_answers

            progress_bar()

    assert len(answers_tokens_probs_map.keys()) == len(question_indexes_unique)

    #

    answers_tokens_indexes_map: Dict[str, np.ndarray] = {}

    ### weights the additional bit probability.
    __weight_answer_probs = lambda answer: answer[0:-1] - answer[-1]

    with alive_bar(len(list(answers_tokens_probs_map.items()))) as progress_bar:
        for (q_index, answers) in answers_tokens_probs_map.items():
            answer_tokens_probs = np.array(
                [__weight_answer_probs(answers[idx]) for idx in range(answers.shape[0])],
                dtype=answers.dtype
            )

            start_index = np.argmax(answer_tokens_probs[:, :, 0].flatten())
            end_index = np.argmax(answer_tokens_probs[:, :, 1].flatten())

            answers_tokens_indexes_map[q_index] = np.array([start_index, end_index], dtype=np.int16)

            progress_bar()

    return answers_tokens_indexes_map


def __compute_answers_predictions(answers_tokens_indexes_map: Any) -> Dict[str, str]:

    answers_for_question_map = {}

    qids, _, passages, _ = FeaturesUtils.QP_data_from_dataset(Dataset.extract("original"))
    qids_unique = list(np.unique(qids))

    passage_by_question_map = {}
    with alive_bar(len(qids_unique)) as progress_bar:
        for qid in qids_unique:
            passage_by_question_map[qid] = np.concatenate((passages[qids == qid])).tolist()
            progress_bar()

    with alive_bar(len(qids_unique)) as progress_bar:
        for qid in qids_unique:
            answer = ""
            passage_tokens = passage_by_question_map[qid]

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

                answer = "".join(passage_tokens[answ_token_start_index:answer_token_end_index + 1]
                                 ).strip()

            answers_for_question_map[qid] = answer

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
    __store_answers_predictions(answers_for_question, "pred")

    print()
    print("The generated answers (json with predictions) file is available at:")
    print(str(LocalStorage.answers_predictions_url("pred")))
    print()

    #

    # Y_true = FeaturesUtils.Y_from_raw_labels(Dataset.extract("labels"))
    # answers_tokens_indexes = __compute_answers_tokens_indexes(Y_true)
    # answers_for_question = __compute_answers_predictions(answers_tokens_indexes)
    # __store_answers_predictions(answers_for_question, "training.true")

    #

    Dataset.optimize_memory()
    Y_pred = None
    answers_tokens_indexes = None
    answers_for_question = None


###

if __name__ == "__main__":
    json_file_url = DataUtils.get_input_file()
    # file_url = DataUtils.get_first_argv()
    Dataset.load(json_path=json_file_url)
    DataReader.model_weights()
    #download weights
    test()
