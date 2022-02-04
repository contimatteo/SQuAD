from typing import List
import numpy as np
import pandas as pd

###


def find_match(passage: List[str], question: List[str]):
    return [1 if p in question else 0 for p in passage]


def original_form(passage: List[str], question: List[str]):
    # exact match in original form
    return find_match(passage, question)


def lower_case_match(passage: List[str], question: List[str]):
    passage = [p.lower() for p in passage]
    question = [p.lower() for p in question]
    return find_match(passage, question)


def passage_question_match(
    passage: List[str], question: List[str], passage_lemm: List[str], question_lem: List[str]
):
    exact = original_form(passage, question)
    lower = lower_case_match(passage, question)
    lemm = lower_case_match(passage_lemm, question_lem)
    return list(zip(exact, lower, lemm))


def apply_exact_match(df: pd.DataFrame):
    df["exact_match"] = df.apply(
        lambda x: passage_question_match(
            x["word_tokens_passage"], x["word_tokens_question"], x["lemmatized_passage"], x[
                "lemmatized_question"]
        ),
        axis=1
    )
    return df
