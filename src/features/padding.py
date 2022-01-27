from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from word_to_index import WordToIndex
MAX_QUESTION_LENGTH = 30
MAX_PASSAGE_LENGTH = 30


def split_in_chunks(row, columns, step):
    new_row = row.copy()
    for col in columns:
        new_row[col] = []
        for i in range(0, len(row[col]), step):
            new_row[col].append(row[col][i:i+step])
    return new_row


def split_passage(df):
    df_clone = df.copy()
    print(df.columns)
    passage_features = ["word_tokens_passage", "word_index_passage", "label", "pos", "pos_onehot",
                        "ner", "ner_onehot", "term_frequency", "exact_match"]
    df_clone = df_clone.apply(lambda x: split_in_chunks(x, passage_features, MAX_PASSAGE_LENGTH), axis=1)

    df_final = df_clone.explode(passage_features)
    return df_final


def apply_padding_to(df: pd.DataFrame, WTI: WordToIndex):
    PAD_WORD = "."
    PAD_WORD_ENCODING = WTI.get_word_index(PAD_WORD)
    df_padded = split_passage(df)
    question_padded = pad_sequences(df_padded['word_index_question'].to_list(), maxlen=MAX_QUESTION_LENGTH, padding="post",
                                 truncating="post", dtype=object, value=PAD_WORD_ENCODING)
    passage_padded = pad_sequences(df_padded['word_index_passage'].to_list(), maxlen=MAX_PASSAGE_LENGTH,
                                    padding="post",
                                    truncating="post", dtype=object, value=PAD_WORD_ENCODING)

    # 'label', 'word_tokens_passage',
    #        'word_tokens_question', 'word_index_passage', 'word_index_question',
    #        'pos', 'pos_onehot', 'ner', 'ner_onehot', 'term_frequency',
    #        'exact_match', 'word_index_question_padded',
    #        'word_index_passage_padded'

    df_padded['word_index_question_padded'] = list(question_padded)
    df_padded['word_index_passage_padded'] = list(passage_padded)
    return df_padded
