import pandas as pd
import numpy as np
from nltk import pos_tag

from .one_hot_encoder import OneHotEncoder

###

pos_tag_dict = {}

###


def delete_cache_pos():
    global pos_tag_dict
    pos_tag_dict = None


def pos_tag_cache(words: str, passage_index: int):
    if passage_index not in pos_tag_dict.keys():
        pos_tag_dict[passage_index] = [p[1] for p in pos_tag(words)]
    return pos_tag_dict[passage_index]


def apply_pos_tag(df: pd.DataFrame):
    df["pos"] = df.apply(
        lambda x: pos_tag_cache(x["word_tokens_passage"], x["passage_index"]), axis=1
    )
    return df


def apply_pos_one_hot(df: pd.DataFrame):
    pos_list = [
        'LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP',
        ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD',
        'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC',
        'CD', 'POS', '#'
    ]
    OHE = OneHotEncoder()
    OHE.fit(pos_list)
    df["pos_categorical"] = df.apply(
        lambda x: np.uint8(OHE.transform_categorical(x["pos"], x["passage_index"])), axis=1
    )
    df["pos_onehot"] = df.apply(
        lambda x: OHE.transform_one_hot(x["pos_categorical"], x["passage_index"]), axis=1
    )
    return df, OHE
