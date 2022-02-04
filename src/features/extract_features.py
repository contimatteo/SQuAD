import pandas as pd
import numpy as np
from nltk import pos_tag
from data.data_preprocessing import data_preprocessing

from .one_hot_encoder import OneHotEncoder
from .exact_match import apply_exact_match
from .lemmatize import apply_lemmatize
from .name_entity_recognition import apply_ner, apply_ner_one_hot
from .term_frequency import apply_term_frequency
from datetime import datetime
from .pos import apply_pos_tag, apply_pos_one_hot

###


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
    # print(ppl)
    return df, OHE_pos, OHE_ner

