from typing import List


class WordToIndex:
    index_dict = {}
    word_dict = {}

    def fit_word(self, token: str):
        if token not in self.index_dict:
            self.index_dict[token] = self.get_index_len()
            self.word_dict[self.index_dict[token]] = token
        return self.index_dict[token]

    def fit_on_list(self, token_list: List[str]):
        return [self.fit_word(token) for token in token_list]

    def get_word_index(self, token: str):
        if token in self.index_dict:
            return self.index_dict[token]
        else:
            return 0

    def get_index_len(self):
        return len(self.index_dict.keys()) + 1

    def index_to_word(self, index: int):
        if index not in self.word_dict.keys():
            raise Exception("index not found")
        return self.word_dict[index]
