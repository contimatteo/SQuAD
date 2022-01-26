import sys
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize
# from nltk import pos_tag
import pandas as pd
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'))
from data_preprocessing import data_preprocessing

lemmatize_passage_dict = {}
lemmatize_question_dict = {}


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('N'):
        return wordnet.NOUN
    if tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def lemmatize(lemmatizer, tokens, passage, pos_tag, lemmatize_dict):
    if passage not in lemmatize_dict.keys():
        token_lemmatized = []
        for word, pos in zip(tokens, pos_tag):
            pos = get_wordnet_pos(pos)
            if pos == '':
                token_lemmatized.append(word)
            else:
                token_lemmatized.append(lemmatizer.lemmatize(word, pos=pos))
        lemmatize_dict[passage] = token_lemmatized
    return lemmatize_dict[passage]


# def lemmatize(df_row, lemmatizer, word_tokens_name):
#     tokens = df_row[word_tokens_name]
#     pos_tag_dict = dict(pos_tag (tokens))
#     token_lemmatized = []
#     for x in tokens:
#         pos = get_wordnet_pos(pos_tag_dict[x])
#         if pos == '':
#             token_lemmatized.append(x)
#         else:
#             token_lemmatized.append(lemmatizer.lemmatize(x, pos = pos))
#     return token_lemmatized


# DoD = {}
# def df_apply_function_with_dict(df, function, dict_key, key, **kwargs):
#    if dict_key not in DoD.keys():
#        DoD[dict_key] = {}
#    dictionary = DoD[dict_key]

#    def __apply(df_row):
#        if df_row[key] not in dictionary.keys():
#            dictionary[df_row[key]] = function(df_row, **kwargs)
#        return dictionary[df_row[key]]

#    return df.apply(lambda x: __apply(x), axis = 1)


def apply_lemmatize(df: pd.DataFrame):
    lemmatizer = WordNetLemmatizer()
    df["lemmatized_passage"] = df.apply(
        lambda x: lemmatize(lemmatizer, x["word_tokens_passage"], x["passage"], x["pos"], lemmatize_passage_dict), axis=1)
    df["lemmatized_question"] = df.apply(
        lambda x: lemmatize(lemmatizer, x["word_tokens_question"], x["passage"], x["pos"], lemmatize_question_dict), axis=1)
    # df["lemmatized_passage"] = df_apply_function_with_dict(df, lemmatize, "lemmatize_passage_dict", "passage", lemmatizer=lemmatizer, word_tokens_name="word_tokens_passage")
    # df["lemmatized_question"] = df_apply_function_with_dict(df, lemmatize, "lemmatize_question_dict", "passage", lemmatizer=lemmatizer, word_tokens_name="word_tokens_question")
    return df


def main_test():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    df_cleaned, glove_matrix, WTI = data_preprocessing()
    print("Applying lemmatization")
    df = apply_lemmatize(df_cleaned[0:5].copy())
    print("Applied lemmatization")
    print(df.columns)

    print(df[0:1])

    # my_fun(arg1 = "pippo", arg2 = "pluto")


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    df_cleaned, glove_matrix, WTI = data_preprocessing()
    print("Applying lemmatization")
    df = apply_lemmatize(df_cleaned)
    print("Applied lemmatization")
    print(df.columns)

    print(df[0:1])


if __name__ == "__main__":
    main_test()
