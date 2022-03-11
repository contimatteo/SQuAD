import numpy as np
import pandas as pd

from nltk import pos_tag
from tensorflow.keras.preprocessing.sequence import pad_sequences

import utils.configs as Configs

from .word_to_index import WordToIndex
from .one_hot_encoder import OneHotEncoder

###


def apply_mask(df: pd.DataFrame):
    df["mask_passage"] = df.apply(
        lambda x: list(np.ones((len(x["word_tokens_passage"]), ), dtype=int)), axis=1
    )
    df["mask_question"] = df.apply(
        lambda x: list(np.ones((len(x["word_tokens_question"]), ), dtype=int)), axis=1
    )
    return df


def split_in_chunks(row, columns, step):
    new_row = row.copy()
    for col in columns:
        new_row[col] = []
        for i in range(0, len(row[col]), step):
            new_row[col].append(row[col][i:i + step])
    return new_row


def split_passage(df):
    df_clone = df.copy()
    passage_features = [
        "word_tokens_passage", "word_index_passage", "pos", "pos_onehot", "ner", "ner_onehot",
        "term_frequency", "exact_match", "mask_passage"
    ]
    if "label" in df:
        passage_features.append("label")
    df_clone = df_clone.apply(
        lambda x: split_in_chunks(x, passage_features, Configs.N_PASSAGE_TOKENS), axis=1
    )

    df_final = df_clone.explode(passage_features)
    return df_final


def pad(df_col, max_seq_len, pad_value):
    return pad_sequences(
        df_col.to_list(),
        maxlen=max_seq_len,
        padding="post",
        truncating="post",
        dtype=object,
        value=pad_value
    )


def apply_padding_to(
    df: pd.DataFrame, wti: WordToIndex, ohe_pos: OneHotEncoder, ohe_ner: OneHotEncoder
):
    PAD_WORD = WordToIndex.PAD_WORD
    # PAD_WORD_ENCODING = wti.get_word_index(PAD_WORD)
    PAD_WORD_ENCODING = 0
    LABEL = (0, 0)
    EXACT_MATCH = (False, False, False)
    POS = pos_tag([PAD_WORD])[0][1]
    POS_CATEGORICAL = ohe_pos.get_categorical_in_dict(POS)
    POS_ONEHOT = ohe_pos.get_ohe_in_dict(POS_CATEGORICAL)
    NER = "O"
    NER_CATEGORICAL = ohe_ner.get_categorical_in_dict(NER)
    NER_ONEHOT = ohe_ner.get_ohe_in_dict(NER_CATEGORICAL)
    TF = 0.0

    df = apply_mask(df)
    df_padded = split_passage(df)
    df_padded["question_index"] = df_padded.index  # df_padded["id"]
    df_padded["chunk_index"] = df_padded.groupby("question_index").cumcount()

    word_index_passage = pad(
        df_padded['word_index_passage'], Configs.N_PASSAGE_TOKENS, PAD_WORD_ENCODING
    )
    word_index_question = pad(
        df_padded['word_index_question'], Configs.N_QUESTION_TOKENS, PAD_WORD_ENCODING
    )
    word_tokens_passage = pad(df_padded['word_tokens_passage'], Configs.N_PASSAGE_TOKENS, PAD_WORD)
    word_tokens_passage_with_spaces = pad(
        df_padded['word_tokens_passage_with_spaces'], Configs.N_PASSAGE_TOKENS, PAD_WORD
    )
    word_tokens_question = pad(
        df_padded['word_tokens_question'], Configs.N_QUESTION_TOKENS, PAD_WORD
    )
    if "label" in df_padded:
        label = pad(df_padded['label'], Configs.N_PASSAGE_TOKENS, LABEL)
    exact_match = pad(df_padded['exact_match'], Configs.N_PASSAGE_TOKENS, EXACT_MATCH)
    pos = pad(df_padded['pos'], Configs.N_PASSAGE_TOKENS, POS)
    pos_categorical = pad(df_padded['pos_categorical'], Configs.N_PASSAGE_TOKENS, POS_CATEGORICAL)
    pos_onehot = pad(df_padded['pos_onehot'], Configs.N_PASSAGE_TOKENS, POS_ONEHOT)
    ner = pad(df_padded['ner'], Configs.N_PASSAGE_TOKENS, NER)
    ner_categorical = pad(df_padded['ner_categorical'], Configs.N_PASSAGE_TOKENS, NER_CATEGORICAL)
    ner_onehot = pad(df_padded['ner_onehot'], Configs.N_PASSAGE_TOKENS, NER_ONEHOT)
    term_frequency = pad(df_padded['term_frequency'], Configs.N_PASSAGE_TOKENS, TF)
    mask_passage = pad(df_padded['mask_passage'], Configs.N_PASSAGE_TOKENS, 0)
    mask_question = pad(df_padded['mask_question'], Configs.N_QUESTION_TOKENS, 0)

    df_padded['word_tokens_passage_padded'] = list(word_tokens_passage)
    df_padded['word_tokens_passage_padded_with_spaces'] = list(word_tokens_passage_with_spaces)
    df_padded['word_index_passage_padded'] = list(word_index_passage)
    df_padded['word_tokens_question_padded'] = list(word_tokens_question)
    df_padded['word_index_question_padded'] = list(word_index_question)

    if "label" in df_padded:
        df_padded['label_padded'] = list(label)
    df_padded['exact_match_padded'] = list(exact_match)

    df_padded['pos_padded'] = list(pos)
    df_padded['pos_categorical_padded'] = list(pos_categorical)
    df_padded['pos_onehot_padded'] = list(pos_onehot)

    df_padded['ner_padded'] = list(ner)
    df_padded['ner_categorical_padded'] = list(ner_categorical)
    df_padded['ner_onehot_padded'] = list(ner_onehot)

    df_padded['term_frequency_padded'] = list(term_frequency)

    df_padded['mask_passage_padded'] = list(mask_passage)
    df_padded['mask_question_padded'] = list(mask_question)
    df_padded = df_padded.reset_index()

    # df_padded.set_index(["passage_index", "question_index", "chunk_index"], drop=False, inplace=True)
    return df_padded
