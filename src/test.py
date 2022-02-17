# pylint: disable=unused-import
from typing import Any, Tuple, List, Dict

import os
import json
from xmlrpc.client import boolean
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
from utils.data import get_argv

###

os.environ["WANDB_JOB_TYPE"] = "training"

set_learning_phase(0)

LocalStorage = LocalStorageManager()

N_ROWS_SUBSET = None  #  `None` for all rows :)

###


def __predict():
    X, _ = X_data_from_dataset(get_data("features"), N_ROWS_SUBSET)
    glove_matrix = get_data("glove")

    model = DRQA(glove_matrix)

    ### load weights
    nn_checkpoint_directory = LocalStorage.nn_checkpoint_url(model.name)
    assert nn_checkpoint_directory.is_file()
    model.load_weights(str(nn_checkpoint_directory))

    Y_pred = model.predict(X, batch_size=128, verbose=1)

    return Y_pred


def __compute_answers_tokens_indexes(Y: np.ndarray,
                                     complementar_bit=False) -> Dict[str, np.ndarray]:
    _, question_indexes = X_data_from_dataset(get_data("features"), N_ROWS_SUBSET)
    question_indexes_unique = list(np.unique(question_indexes))

    answers_tokens_probs_map = {}

    with alive_bar(len(question_indexes_unique)) as progress_bar:
        for qid in question_indexes_unique:
            subset_indexes = np.array(question_indexes == qid)
            current_answers = Y[subset_indexes]
            answers_tokens_probs_map[qid] = current_answers

            progress_bar()

    assert len(answers_tokens_probs_map.keys()) == len(question_indexes_unique)

    # test_qid = "5733bf84d058e614000b61c1"
    # whole_passage_answer_place = np.vstack(tuple(answers_tokens_probs_map[test_qid].tolist()))

    # print()
    # print()
    # print(whole_passage_answer_place)
    # print()

    # start_index = -1
    # end_index = -1
    # try:
    #     start_index = np.where(np.all(whole_passage_answer_place == np.array([1, 0]),
    #                                   axis=1))[0].tolist()[0]
    #     end_index = np.where(np.all(whole_passage_answer_place == np.array([0, 1]),
    #                                 axis=1))[0].tolist()[0]
    # except:
    #     start_index = np.where(np.all(whole_passage_answer_place == np.array([1, 1]),
    #                                   axis=1))[0].tolist()[0]
    #     end_index = start_index

    # print()
    # print(start_index)
    # print(end_index)
    # print()

    #

    __weight_answer_probs = lambda answer: answer[0:1]

    if complementar_bit:
        __weight_answer_probs = lambda answer: answer[0:-1] - answer[-1]

    answers_tokens_indexes_map: Dict[str, np.ndarray] = {}

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

    #
    # print()
    # print("NEW STEP")
    # print(answers_tokens_indexes_map[test_qid])
    # print()
    # raise Exception("stop")
    return answers_tokens_indexes_map


def __compute_answers_predictions(answers_tokens_indexes_map: Any) -> Dict[str, str]:

    answers_for_question_map = {}

    qids, _, passages = QP_data_from_dataset(get_data("original"))
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

                # answ_span_pre_start = passage_tokens[0:answ_token_start_index]
                # answ_span_to_end = passage_tokens[0:answer_token_end_index + 1]
                # ### INFO: we have always to consider the ENTIRE end token.
                # answ_span_to_end += [str(passage_tokens[answer_token_end_index])]
                # ### compute the chars range indexes
                # answ_char_start_index = len(" ".join(answ_span_pre_start))
                # answ_char_end_index = len(" ".join(answ_span_to_end))
                # ### extract answer from chars indexes
                # answer = passage[answ_char_start_index:answ_char_end_index]

                unicode_answer_list = [x for x in passage_tokens[answ_token_start_index:answer_token_end_index + 1] if not x.isascii()]

                answer = "".join(
                    passage_tokens[answ_token_start_index:answer_token_end_index + 1]
                ).strip()

                # answer = answer.replace(" ' ", "'")
                # answer = answer.replace(" - ", "-")


                # for unicode in unicode_answer_list:
                #     answer = answer.replace(" "+unicode, ""+unicode).replace(unicode+" ", unicode+"")



            # if qid in answers_tokens_indexes_map:
            #     answers_for_question_map[qid] = answer
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
    answers_tokens_indexes = __compute_answers_tokens_indexes(Y_pred, complementar_bit=True)
    answers_for_question = __compute_answers_predictions(answers_tokens_indexes)
    __store_answers_predictions(answers_for_question, "training.pred")

    print()
    print("The generated answers (json with predictions) file is available at:")
    print(str(LocalStorage.answers_predictions_url("training.pred")))
    print()

    #

    ### TODO: remove the following code ...
    Y_true = Y_data_from_dataset(get_data("labels"), N_ROWS_SUBSET)
    if not Configs.COMPLEMENTAR_BIT:
        Y_true = Y_true[:, :Configs.N_PASSAGE_TOKENS, :]

    ### TODO: remove the following code ...
    answers_tokens_indexes = __compute_answers_tokens_indexes(Y_true, complementar_bit=True)
    answers_for_question = __compute_answers_predictions(answers_tokens_indexes)
    __store_answers_predictions(answers_for_question, "training.true")


###

if __name__ == "__main__":
    # load_data(json_path="./data/raw/train.v3.json")

    load_data(json_path="./data/raw/train.v7.json")

    # load_data()

    test()
