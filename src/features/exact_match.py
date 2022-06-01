from typing import List

import pandas as pd

###


class ExactMatchFeature:

    @staticmethod
    def __find_match(passage: List[str], question: List[str]):
        return [1 if p in question else 0 for p in passage]

    @staticmethod
    def __original_form(passage: List[str], question: List[str]):
        ### exact match in original form
        return ExactMatchFeature.__find_match(passage, question)

    @staticmethod
    def __lower_case_match(passage: List[str], question: List[str]):
        passage = [p.lower() for p in passage]
        question = [p.lower() for p in question]

        return ExactMatchFeature.__find_match(passage, question)

    @staticmethod
    def __passage_question_match(
        passage: List[str], question: List[str], passage_lemm: List[str], question_lem: List[str]
    ):
        exact = ExactMatchFeature.__original_form(passage, question)
        lower = ExactMatchFeature.__lower_case_match(passage, question)
        lemm = ExactMatchFeature.__lower_case_match(passage_lemm, question_lem)

        return list(zip(exact, lower, lemm))

    #

    @staticmethod
    def apply_to_df(df: pd.DataFrame):
        df["exact_match"] = df.apply(
            lambda x: ExactMatchFeature.__passage_question_match(
                x["word_tokens_passage"], x["word_tokens_question"], x["lemmatized_passage"], x[
                    "lemmatized_question"]
            ),
            axis=1
        )
        return df
