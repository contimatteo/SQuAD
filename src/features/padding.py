from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from word_to_index import WordToIndex
from one_hot_encoder import OneHotEncoder
from nltk import pos_tag

MAX_QUESTION_LENGTH = 30
MAX_PASSAGE_LENGTH = 100


def split_in_chunks(row, columns, step):
    new_row = row.copy()
    for col in columns:
        new_row[col] = []
        for i in range(0, len(row[col]), step):
            new_row[col].append(row[col][i:i+step])
    return new_row


def split_passage(df):
    df_clone = df.copy()
    # print(df.columns)
    passage_features = ["word_tokens_passage", "word_index_passage", "label", "pos", "pos_onehot",
                        "ner", "ner_onehot", "term_frequency", "exact_match"]
    df_clone = df_clone.apply(lambda x: split_in_chunks(x, passage_features, MAX_PASSAGE_LENGTH), axis=1)

    df_final = df_clone.explode(passage_features)
    return df_final


def apply_padding_to(df: pd.DataFrame, WTI: WordToIndex, OHE_pos: OneHotEncoder, OHE_ner: OneHotEncoder ):
    PAD_WORD = "."
    PAD_WORD_ENCODING = WTI.get_word_index(PAD_WORD)
    LABEL = (0, 0)
    EXACT_MATCH = (False, False, False)
    POS = pos_tag([PAD_WORD])[0][1]
    POS_ONEHOT = OHE_pos.get_OHE_in_dict(POS)
    NER = "O"
    NER_ONEHOT = OHE_ner.get_OHE_in_dict(NER)
    TF = 0.0
    df_padded = split_passage(df)

    word_index_passage = pad_sequences(df_padded['word_index_passage'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                                       padding="post",
                                       truncating="post", dtype=object, value=PAD_WORD_ENCODING)
    word_index_question = pad_sequences(df_padded['word_index_question'].to_list(), maxlen=MAX_QUESTION_LENGTH,
                                        padding="post",
                                        truncating="post", dtype=object, value=PAD_WORD_ENCODING)

    word_tokens_passage = pad_sequences(df_padded['word_tokens_passage'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                                        padding="post",
                                        truncating="post", dtype=object, value=PAD_WORD)
    word_tokens_question = pad_sequences(df_padded['word_tokens_question'].to_list(), maxlen=MAX_QUESTION_LENGTH,
                                         padding="post",
                                         truncating="post", dtype=object, value=PAD_WORD)

    label = pad_sequences(df_padded['label'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                          padding="post",
                          truncating="post", dtype=object, value=LABEL)
    exact_match = pad_sequences(df_padded['exact_match'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                                padding="post",
                                truncating="post", dtype=object, value=EXACT_MATCH)

    pos = pad_sequences(df_padded['pos'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                        padding="post",
                        truncating="post", dtype=object, value=POS)
    pos_onehot = pad_sequences(df_padded['pos_onehot'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                               padding="post",
                               truncating="post", dtype=object, value=POS_ONEHOT)

    ner = pad_sequences(df_padded['ner'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                        padding="post",
                        truncating="post", dtype=object, value=NER)
    ner_onehot = pad_sequences(df_padded['ner_onehot'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                               padding="post",
                               truncating="post", dtype=object, value=[NER_ONEHOT])

    term_frequency = pad_sequences(df_padded['term_frequency'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                                   padding="post",
                                   truncating="post", dtype=object, value=TF)

    # df_padded['word_index_passage_padded'] = list(word_index_passage)
    # df_padded['word_index_question_padded'] = list(word_index_question)
    #
    # df_padded['word_tokens_passage_padded'] = list(word_tokens_passage)
    # df_padded['word_tokens_question_padded'] = list(word_tokens_question)
    #
    # df_padded['label_padded'] = list(label)
    # df_padded['exact_match_padded'] = list(exact_match)
    #
    # df_padded['pos_padded'] = list(pos)
    # df_padded['pos_onehot_padded'] = list(pos_onehot)
    #
    # df_padded['ner_padded'] = list(ner)
    # df_padded['ner_onehot_padded'] = list(ner_onehot)
    #
    # df_padded['term_frequency_padded'] = list(term_frequency)

    df_padded['word_tokens_passage_padded'] = list(word_tokens_passage)
    df_padded['word_index_passage_padded'] = list(word_index_passage)
    df_padded['word_tokens_question_padded'] = list(word_tokens_question)
    df_padded['word_index_question_padded'] = list(word_index_question)

    df_padded['label_padded'] = list(label)
    df_padded['exact_match_padded'] = list(exact_match)

    df_padded['pos_padded'] = list(pos)
    df_padded['pos_onehot_padded'] = list(pos_onehot)

    df_padded['ner_padded'] = list(ner)
    df_padded['ner_onehot_padded'] = list(ner_onehot)

    df_padded['term_frequency_padded'] = list(term_frequency)

    # df_padded.set_index(["passage_index"], inplace=True)

    return df_padded
