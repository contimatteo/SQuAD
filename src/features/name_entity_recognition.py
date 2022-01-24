import sys
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize
# from nltk import pos_tag
import pandas as pd
from nltk import ne_chunk
import os
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'))
from data_preprocessing import data_preprocessing
from lemmatize import apply_lemmatize


def ner(pos):
    ne_tree = ne_chunk(pos)
    iob_tagged = tree2conlltags(ne_tree)
    return [iob[-1] for iob in iob_tagged]


def apply_ner(df: pd.DataFrame):
    df["ner"] = df.apply(
        lambda x: ner(list(zip(x["word_tokens_passage"], x["pos"]))), axis=1)
    return df
