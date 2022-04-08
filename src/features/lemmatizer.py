import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

###

lemmatize_passage_dict = {}
lemmatize_question_dict = {}

###


class Lemmatizer:

    @staticmethod
    def __get_wordnet_pos(tag):
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

    @staticmethod
    def __lemmatize_tokens(lemmatizer, tokens, pos_tag, lemmatize_dict, sentence):
        if sentence not in lemmatize_dict.keys():
            token_lemmatized = []
            for word, pos in zip(tokens, pos_tag):
                pos = Lemmatizer.__get_wordnet_pos(pos)
                if pos == '':
                    token_lemmatized.append(word)
                else:
                    token_lemmatized.append(lemmatizer.lemmatize(word, pos=pos))
            lemmatize_dict[sentence] = token_lemmatized
        return lemmatize_dict[sentence]

    #

    @staticmethod
    def apply_to_df(df: pd.DataFrame):
        lemmatizer = WordNetLemmatizer()

        df["lemmatized_passage"] = df.apply(
            lambda x: Lemmatizer.__lemmatize_tokens(
                lemmatizer, x["word_tokens_passage"], x["pos"], lemmatize_passage_dict, x["passage"]
            ),
            axis=1
        )
        df["lemmatized_question"] = df.apply(
            lambda x: Lemmatizer.__lemmatize_tokens(
                lemmatizer, x["word_tokens_question"], x["pos"], lemmatize_question_dict, x[
                    "question"]
            ),
            axis=1
        )

        return df

    @staticmethod
    def delete_cache():
        global lemmatize_passage_dict, lemmatize_question_dict

        lemmatize_passage_dict = None
        lemmatize_question_dict = None
