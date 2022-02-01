import pandas as pd

from nltk import pos_tag
from data.data_preprocessing import data_preprocessing

from .one_hot_encoder import OneHotEncoder
from .exact_match import apply_exact_match
from .lemmatize import apply_lemmatize
from .name_entity_recognition import apply_ner, apply_ner_one_hot
from .term_frequency import apply_term_frequency

###

pos_tag_dict = {}

###


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
    df["pos_onehot"] = df.apply(lambda x: OHE.transform(x["pos"], x["passage_index"]), axis=1)
    return df, OHE


def drop_useless_columns(df: pd.DataFrame):
    useless_columns = ["passage", "question", "lemmatized_passage", "lemmatized_question"]
    df.drop(useless_columns, axis=1, inplace=True)
    return df


def extract_features(df: pd.DataFrame):
    print("Applying POS")
    df = apply_pos_tag(df)
    df, OHE_pos = apply_pos_one_hot(df)
    print("Applying NER")
    df = apply_ner(df)
    df, OHE_ner = apply_ner_one_hot(df)
    print("Applying TF")
    df = apply_term_frequency(df)
    print("Applying LEMMATIZATION")
    df = apply_lemmatize(df)
    print("Applying EXACT MATCH")
    df = apply_exact_match(df)
    df = drop_useless_columns(df)
    return df, OHE_pos, OHE_ner

