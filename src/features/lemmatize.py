import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from data.data_preprocessing import data_preprocessing

###

lemmatize_passage_dict = {}
lemmatize_question_dict = {}

###


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


def lemmatize(lemmatizer, tokens, pos_tag, lemmatize_dict, sentence):
    if sentence not in lemmatize_dict.keys():
        token_lemmatized = []
        for word, pos in zip(tokens, pos_tag):
            pos = get_wordnet_pos(pos)
            if pos == '':
                token_lemmatized.append(word)
            else:
                token_lemmatized.append(lemmatizer.lemmatize(word, pos=pos))
        lemmatize_dict[sentence] = token_lemmatized
    return lemmatize_dict[sentence]


def apply_lemmatize(df: pd.DataFrame):
    lemmatizer = WordNetLemmatizer()
    df["lemmatized_passage"] = df.apply(
        lambda x: lemmatize(
            lemmatizer, x["word_tokens_passage"], x["pos"], lemmatize_passage_dict, x["passage"]
        ),
        axis=1
    )
    df["lemmatized_question"] = df.apply(
        lambda x: lemmatize(
            lemmatizer, x["word_tokens_question"], x["pos"], lemmatize_question_dict, x["question"]
        ),
        axis=1
    )
    # df["lemmatized_passage"] = df_apply_function_with_dict(df, lemmatize, "lemmatize_passage_dict", "passage", lemmatizer=lemmatizer, word_tokens_name="word_tokens_passage")
    # df["lemmatized_question"] = df_apply_function_with_dict(df, lemmatize, "lemmatize_question_dict", "passage", lemmatizer=lemmatizer, word_tokens_name="word_tokens_question")
    return df

