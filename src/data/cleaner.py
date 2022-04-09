from typing import Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer

from utils import NltkUtils

###

span_tokenize_dict = {}
sentence_tokenize_dict = {}
passage_index_dict = {}

###

# def tokenize_sentence_df(df_row, sentence_name):
#     sentence = df_row[sentence_name]
#     t1, _ = __tokenizers()
#     return t1.tokenize(sentence)


def __tokenizers():
    tokenizer1 = RegexpTokenizer(r'\d+[.,]\d+\b\.*|[A-Z][.A-Z]+\b\.*|\w+|\S')
    tokenizer2 = RegexpTokenizer(r'\d+[.,]\d+\b\.*|[A-Z][.A-Z]+\b\.*|\w+|\S|.')
    return tokenizer1, tokenizer2


def __group_tokens(t, t_with_spaces):
    t1 = deepcopy(t)
    first_item_found = False
    first_string = ""
    j = 0
    for el in t_with_spaces:
        correspondence_not_found = False
        if j < len(t1):
            if t1[j] == el:
                if not first_item_found:
                    t1[j] = first_string + t1[j]
                    first_item_found = True
                j += 1
            else:
                correspondence_not_found = True
        else:
            correspondence_not_found = True

        if correspondence_not_found:
            if first_item_found:
                t1[j - 1] = t1[j - 1] + el
            else:
                first_string += el

    return t1


def __tokenize_sentence(sentence):
    t1, _ = __tokenizers()
    if sentence not in sentence_tokenize_dict.keys():
        sentence_tokenize_dict[sentence] = t1.tokenize(sentence)

    return sentence_tokenize_dict[sentence]


def __tokenize_with_spaces(sentence):
    _, t2 = __tokenizers()

    sentence_tokenized = __tokenize_sentence(sentence)
    sentence_tokenized_with_spaces = t2.tokenize(sentence)
    t_grouped = __group_tokens(sentence_tokenized, sentence_tokenized_with_spaces)
    return t_grouped


def __add_split_into_words(df):
    df["word_tokens_passage"] = df.apply(lambda x: __tokenize_sentence(x["passage"]), axis=1)
    df["word_tokens_passage_with_spaces"] = df.apply(
        lambda x: __tokenize_with_spaces(x["passage"]), axis=1
    )
    df["word_tokens_question"] = df.apply(lambda x: __tokenize_sentence(x["question"]), axis=1)
    return df


def __span_tokenize(sentence, *_):
    tokenized_sentence = __tokenize_with_spaces(sentence)
    span_list = []
    j = 0
    for el in tokenized_sentence:
        span_list.append((j, j + len(el) - 1))
        j += len(el)
    return span_list


def __get_word_pstart_pend(interval: Tuple[int, int], dim: int):
    p_start, p_end = interval
    start = np.zeros(dim, dtype=int)
    end = np.zeros(dim, dtype=int)
    start[p_start] = 1
    end[p_end] = 1
    return list(zip(np.uint8(start), np.uint8(end)))


def __get_answer_start_end(passage, answer_text, answer_start):
    answer_end = len(answer_text) + answer_start - 1

    if passage not in span_tokenize_dict.keys():
        span_tokenize_dict[passage] = __span_tokenize(passage)

    interval = []
    if answer_end + 1 < len(passage):
        if passage[answer_end + 1] == ' ':
            answer_end += 1

    app_dict = {}
    for i, (s, e) in enumerate(span_tokenize_dict[passage]):
        if e >= answer_start and s <= answer_end:
            interval.append(i)
            app_dict[i] = (s, e)

    if len(interval) < 1:
        at = [answer_text]
        print(at)
        return [-1, -1]

    return __get_word_pstart_pend((min(interval), max(interval)), len(span_tokenize_dict[passage]))


def __add_labels(df):
    df["label"] = df.apply(
        lambda x: __get_answer_start_end(x["passage"], x["answer"], x["answer_start"]), axis=1
    )
    return df


def __get_passage_index(passage: str):
    if passage not in passage_index_dict.keys():
        passage_index_dict[passage] = len(passage_index_dict.keys())
    return passage_index_dict[passage]


def __add_passage_index(df: pd.DataFrame):
    df["passage_index"] = df.apply(lambda x: np.uint32(__get_passage_index(x["passage"])), axis=1)
    return df


###


def delete_cache_cleaner():
    global span_tokenize_dict, sentence_tokenize_dict, passage_index_dict

    span_tokenize_dict = None
    sentence_tokenize_dict = None
    passage_index_dict = None


def clean_data(df: pd.DataFrame):
    NltkUtils.download_utilities()

    print("Data cleaning")
    df = __add_passage_index(df)
    if "answer" in df:
        df = __add_labels(df).drop(axis=1, columns='answer_start')

    df = __add_split_into_words(df)
    print("Data cleaned \n")

    return df
