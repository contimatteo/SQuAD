from typing import Tuple

import numpy as np
import pandas as pd

from copy import deepcopy
from nltk.tokenize import RegexpTokenizer

from utils.data import nltk_download_utilities

###

span_tokenize_dict = {}
sentence_tokenize_dict = {}
passage_index_dict = {}

###


def delete_cache_data_cleaning():
    global span_tokenize_dict, sentence_tokenize_dict, passage_index_dict
    span_tokenize_dict = None
    sentence_tokenize_dict = None
    passage_index_dict = None


# def __regex_separator(text,separator):
#   # separator =["�"]#["�"]
#    for sep in separator:
#       text= text.replace(sep," ")
#    return text
#
# def separate_words(df,separator=["-"]):
#    columns=["passage","answer","question"]
#    for col in columns:
#        df[col] = df.apply(lambda x: __regex_separator(x[col],separator), axis = 1)
#    return df


def tokenizers():
    # r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
    tokenizer1 = RegexpTokenizer(r'\d+[.,]\d+\b\.*|[A-Z][.A-Z]+\b\.*|\w+|\S')  # |[A-Z][.A-Z]+\b\.*|
    tokenizer2 = RegexpTokenizer(r'\d+[.,]\d+\b\.*|[A-Z][.A-Z]+\b\.*|\w+|\S|.')
    return tokenizer1, tokenizer2


def group_tokens(t, t_with_spaces):
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


def tokenize_sentence_df(df_row, sentence_name):
    sentence = df_row[sentence_name]
    t1, _ = tokenizers()
    return t1.tokenize(sentence)


def tokenize_sentence(sentence):
    t1, _ = tokenizers()
    if sentence not in sentence_tokenize_dict.keys():
        sentence_tokenize_dict[sentence] = t1.tokenize(sentence)

    return sentence_tokenize_dict[sentence]


# def tokenize_sentence(sentence):
#    t1,_ = tokenizers()
#    if sentence not in get_dict()["sentence_tokenize_dict"].keys():
#        get_dict()["sentence_tokenize_dict"][sentence] = t1.tokenize(sentence)
#
#    return get_dict()["sentence_tokenize_dict"][sentence]


def tokenize_with_spaces(sentence):
    _, t2 = tokenizers()

    sentence_tokenized = tokenize_sentence(sentence)
    sentence_tokenized_with_spaces = t2.tokenize(sentence)
    t_grouped = group_tokens(sentence_tokenized, sentence_tokenized_with_spaces)
    return t_grouped


# def split_into_words(df):
#     df["word_tokens_passage"] = df_apply_function_with_dict(df,tokenize_sentence_df,"sentence_tokenize_dict","passage",sentence_name="passage")
#     df["word_tokens_question"] = df_apply_function_with_dict(df, tokenize_sentence_df,"sentence_tokenize_dict","question",sentence_name="question")
#    return df


def add_split_into_words(df):
    df["word_tokens_passage"] = df.apply(lambda x: tokenize_sentence(x["passage"]), axis=1)
    df["word_tokens_question"] = df.apply(lambda x: tokenize_sentence(x["question"]), axis=1)
    return df


def span_tokenize(sentence, *_):
    tokenized_sentence = tokenize_with_spaces(sentence)
    span_list = []
    j = 0
    for el in tokenized_sentence:
        span_list.append((j, j + len(el) - 1))
        j += len(el)
    return span_list


def get_word_pstart_pend(interval: Tuple[int, int], dim: int):
    p_start, p_end = interval
    start = np.zeros(dim, dtype=int)
    end = np.zeros(dim, dtype=int)
    start[p_start] = 1
    end[p_end] = 1
    return list(zip(np.uint8(start), np.uint8(end)))


def get_answer_start_end(passage, answer_text, answer_start):
    answer_end = len(answer_text) + answer_start - 1

    if passage not in span_tokenize_dict.keys():
        span_tokenize_dict[passage] = span_tokenize(passage)

    # interval = [
    #     i for i, (s, e) in enumerate(span_tokenize_dict[passage])
    #     if s >= answer_start and e <= answer_end
    # ]
    interval = []
    if answer_end + 1 < len(passage):
        # print(passage[answer_start:answer_end + 2])
        if passage[answer_end+1] == ' ':
            answer_end += 1
    app_dict = {}
    for i, (s, e) in enumerate(span_tokenize_dict[passage]):
        if e >= answer_start and s <= answer_end:  # (e == answer_end or e == answer_end - 1):
            interval.append(i)
            app_dict[i] = (s, e)
            # print()
            # print(passage[s:e + 1])
            # print("answer_end - e: ", (answer_end - e))
            # print()

    if len(interval) < 1:
        # raise Exception(interval + " is empty.")
        at = [answer_text]  # [str(passage)[96]]
        print(at)
        return [-1, -1]

    # credi = get_word_pstart_pend((min(interval), max(interval)), len(span_tokenize_dict[passage]))
    #
    #  # (1, 0)(0, 1)
    # # (1, 1)
    #
    # if (1, 0) in credi:
    #     token_start_index = credi.index((1, 0))
    #     token_end_index = credi.index((0, 1))
    #
    # else:
    #     token_start_index = credi.index((1, 1))
    #     token_end_index = token_start_index
    #
    # print()
    # print("MY OUTPUT")
    #
    # s = app_dict[token_start_index]
    # e = app_dict[token_end_index]
    #
    # print(token_start_index)
    # print(token_end_index)
    # print(passage[s[0]:e[1] + 1])
    # print()

    return get_word_pstart_pend((min(interval), max(interval)), len(span_tokenize_dict[passage]))


def add_labels(df):
    df["label"] = df.apply(
        lambda x: get_answer_start_end(x["passage"], x["answer"], x["answer_start"]), axis=1
    )
    return df


def get_passage_index(passage: str):
    if passage not in passage_index_dict.keys():
        passage_index_dict[passage] = len(passage_index_dict.keys())
    return passage_index_dict[passage]


def add_passage_index(df: pd.DataFrame):
    df["passage_index"] = df.apply(lambda x: np.uint32(get_passage_index(x["passage"])), axis=1)
    return df


def data_cleaning(df: pd.DataFrame):
    # df = separate_words(df)
    nltk_download_utilities()
    print()
    print("Data cleaning")
    df = add_passage_index(df)
    if "answer" in df:
        df = add_labels(df).drop(axis=1, columns='answer_start')

    df = add_split_into_words(df)
    print("Data cleaned \n")
    return df
