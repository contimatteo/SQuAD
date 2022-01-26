from typing import List, Tuple

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from copy import deepcopy
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'utils'))
# from preprocessing_utils import df_apply_function_with_dict
# from preprocessing_utils import df_apply_function_with_dict_2
from preprocessing_utils import get_dict, insert_dict
from data_utils import nltk_download_utilities

span_tokenize_dict = {}
sentence_tokenize_dict = {}
passage_index_dict = {}


# def __regex_separator(text,separator):
#   # separator =["�"]#["�"]
#    for sep in separator:
#       text= text.replace(sep," ")
#    return text
#
#
# def separate_words(df,separator=["-"]):
#    columns=["passage","answer","question"]
#    for col in columns:
#        df[col] = df.apply(lambda x: __regex_separator(x[col],separator), axis = 1)
#    return df


def tokenizers():
    # r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'
    tokenizer1 = RegexpTokenizer(r'\d[.,]\d+|\w+|\S')  # |[A-Z][.A-Z]+\b\.*|
    tokenizer2 = RegexpTokenizer(r'\d[.,]\d+|\w+|\S|.')
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
                t1[j-1] = t1[j-1] + el
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
    return list(zip(start, end))


def get_answer_start_end(passage, answer_text, answer_start):
    answer_end = len(answer_text) + answer_start

    if passage not in span_tokenize_dict.keys():
        span_tokenize_dict[passage] = span_tokenize(passage)

    interval = [i for i, (s, e) in enumerate(span_tokenize_dict[passage]) if e >= answer_start and s <= answer_end]
    if len(interval) < 1:
        # raise Exception(interval + " is empty.")
        at = [answer_text]  # [str(passage)[96]]
        print(at)
        return [-1, -1]

    return get_word_pstart_pend((min(interval), max(interval)), len(span_tokenize_dict[passage]))


def add_labels(df):
    df["label"] = df.apply(lambda x: get_answer_start_end(x["passage"], x["answer"], x["answer_start"]), axis=1)
    return df


# def get_answer_start_end(df_row,passage_name,answer_name,answer_start_name):
#    passage=df_row[passage_name]
#    answer_text=df_row[answer_name]
#    answer_start=df_row[answer_start_name]
#
#    answer_end = len(answer_text) + answer_start
#    interval = [i for i, (s, e) in enumerate(span_tokenize(passage)) if e >= answer_start and s <= answer_end]
#    if len(interval) <1:
#       #raise Exception(interval + " is empty.")
#       err= [answer_text]#[str(passage)[96]]
#       print("anwer not found: ",err)
#       return [-1,-1]
#    return [min(interval),max(interval)]
#
#
# def add_labels(df):
#    df["label"] = df_apply_function_with_dict_2(df, get_answer_start_end,span_tokenize,"span_tokenize_dict","passage",passage_name="passage",answer_name="answer",answer_start_name="answer_start")
#    #df.apply(lambda x: get_answer_start_end(x["passage"], x["answer"],x["answer_start"]), axis = 1)
#    return df


def get_passage_index(passage: str):
    if passage not in passage_index_dict.keys():
        passage_index_dict[passage] = len(passage_index_dict.keys())
    return passage_index_dict[passage]


def add_passage_index(df: pd.DataFrame):
    df["passage_index"] = df.apply(lambda x: get_passage_index(x["passage"]), axis=1)
    return df


def data_cleaning(df: pd.DataFrame):
    # df = separate_words(df)
    nltk_download_utilities()
    print()
    print("Data cleaning")
    df = add_passage_index(df)
    df = add_labels(df).drop(axis=1, columns='answer_start')
    df = add_split_into_words(df)
    print("Data cleaned \n")
    return df


def main():
    insert_dict("sentence_tokenize_dict")
    print(get_dict())
    s = "   anna.    va  "
    nltk_download_utilities()
    print(tokenize_with_spaces(s))
    print(span_tokenize(s))
    print(get_dict())


if __name__ == "__main__":
    main()
