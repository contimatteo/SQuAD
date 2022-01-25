import sys
import pandas as pd
import os
from typing import List


# class WordCounting:
#     passage_dict = {}
#     word_count_dict = {}
#     total_word_num = 0
#
#     def get_unique_passages(self, passage: str, token_passage: List[str]):
#         if passage not in self.passage_dict.keys():
#             self.passage_dict[passage] = token_passage
#
#     def count_words(self, df: pd.DataFrame):
#         df.apply(lambda x: self.get_unique_passages(x["passage"], x["word_tokens_passage"]), axis=1)
#         total_sum = 0
#         for token_list in self.passage_dict.values():
#             total_sum += len(token_list)
#             for token in token_list:
#                 if token not in self.word_count_dict.keys():
#                     self.word_count_dict[token] = 1
#                 else:
#                     self.word_count_dict[token] += 1
#         self.total_word_num += total_sum
#
#     def get_term_freq_normalized(self, tokens: List[str]):
#         return [(self.word_count_dict[word] / self.total_word_num) for word in tokens]


def get_term_freq_normalized(token_list: List[str]):
    word_count_dict = {}
    total_sum = len(token_list)
    for token in token_list:
        if token not in word_count_dict.keys():
            word_count_dict[token] = 1
        else:
            word_count_dict[token] += 1
    tf = []
    for token in token_list:
        tf.append(word_count_dict[token] / total_sum)
    return tf


# WC = WordCounting()


def apply_term_frequency(df: pd.DataFrame, is_training=True):
    # if is_training:
    # WC.count_words(df)
    df["term_frequency"] = df.apply(lambda x: get_term_freq_normalized(x["word_tokens_passage"]), axis=1)
    # df["term_frequency"] = df.apply(lambda x: WC.get_term_freq_normalized(x["word_tokens_passage"]), axis=1)
    return df

