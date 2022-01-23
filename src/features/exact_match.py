import sys
from typing import List
import pandas as pd
from lemmatize import apply_lemmatize
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'))
from data_preprocessing import data_preprocessing

# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'utils'))
from preprocessing_utils import df_apply_function_with_dict


def find_match(passage: List[str], question: List[str]):
    return [p in question for p in passage]


def original_form(passage: List[str], question: List[str]):
    # exact match in original form

    return find_match(passage, question)


def lower_case_match(passage: List[str], question: List[str]):
    passage = [p.lower() for p in passage]
    question = [p.lower() for p in question]
    return find_match(passage, question)


def passage_question_match(passage: List[str], question: List[str], passage_lemm: List[str], question_lem: List[str]):
    exact = original_form(passage, question)
    lower = lower_case_match(passage, question)
    lemm = lower_case_match(passage_lemm, question_lem)
    return list(zip(exact, lower, lemm))


def apply_exact_match(df: pd.DataFrame):
    df["exact_match"] = df.apply(
        lambda x: passage_question_match(x["word_tokens_passage"], x["word_tokens_question"],
                                         x["lemmatized_passage"], x["lemmatized_question"]), axis=1)
    return df


def exact_match(df: pd.DataFrame):
    print("Applying lemmatization")
    df = apply_lemmatize(df)
    print("Applied lemmatization")
    print("Applying exact match")
    df = apply_exact_match(df)
    print("Applied exact match")
    return df


# def mayfun2(arg1, arg2):
#     print(arg1)
#     print(arg2)
# 
#
# def my_fun(**kwargs):
#     mayfun2(**kwargs)


def main_test():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    df_cleaned = data_preprocessing()
    df = exact_match(df_cleaned[0:5].copy())
    print(df.columns)

    print(df[0:1])

    # my_fun(arg1 = "pippo", arg2 = "pluto")


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    df_cleaned = data_preprocessing()
    df = exact_match(df_cleaned)
    print(df.columns)

    print(df[0:1])


if __name__ == "__main__":
    main_test()
