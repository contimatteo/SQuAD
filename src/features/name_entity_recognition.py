import sys
from typing import Union, List
import pandas as pd
from nltk import ne_chunk
import os
from nltk.chunk import conlltags2tree, tree2conlltags, ieerstr2tree
from pprint import pprint
from one_hot_encoder import OneHotEncoder
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'))
from data_preprocessing import data_preprocessing
from lemmatize import apply_lemmatize

ner_dict = {}


def ner(pos, passage_index: int):
    if passage_index not in ner_dict.keys():
        ne_tree = ne_chunk(pos)
        iob_tagged = tree2conlltags(ne_tree)
        ner_dict[passage_index] = [iob[-1] for iob in iob_tagged]
    return ner_dict[passage_index]


def apply_ner(df: pd.DataFrame):
    df["ner"] = df.apply(
        lambda x: ner(list(zip(x["word_tokens_passage"], x["pos"])), x["passage_index"]), axis=1)
    return df


def apply_ner_one_hot(df: pd.DataFrame):
    ner_special = ["O"]
    ner_entities = ["FACILITY", "GPE", "GSP", "LOCATION", "ORGANIZATION", "PERSON"]
    ner_prefix = ["B-", "I-"]
    ner_list = ner_special
    for ent in ner_entities:
        for pref in ner_prefix:
            ner_list.append(pref + ent)
    OHE = OneHotEncoder()
    OHE.fit(ner_list)
    df["ner_onehot"] = df.apply(
        lambda x: [p for p in OHE.transform(x["ner"], x["passage_index"])], axis=1)
    return df

