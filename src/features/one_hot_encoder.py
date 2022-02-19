from typing import List

import numpy as np

###


class OneHotEncoder:
    # one_hot_dict = {}
    # cache_dict = {}

    def fit(self, pos_list):
        # self.one_hot_dict = {}
        pos_len = len(pos_list)

        id_m = np.identity(pos_len, dtype="int8")
        pos_dict = {}
        for i, el in enumerate(pos_list):
            # i = np.uint8(i)
            self.categorical_dict[el] = i
            self.one_hot_dict[i] = id_m[i]
        return self.one_hot_dict

    def get_one_hot_dict(self):
        return self.one_hot_dict

    def get_categorical_dict(self):
        return self.categorical_dict

    def transform_categorical(self, df_row: List[str], passage_index: int):
        if passage_index not in self.cache_categorical_dict.keys():
            return_list = []
            for el in df_row:
                if el in self.categorical_dict.keys():
                    return_list.append(self.categorical_dict[el])
                else:
                    return_list.append(None)
                    print(f"unable to encode onehot of NER/POS tag: {el}")
            self.cache_categorical_dict[passage_index] = return_list
        return self.cache_categorical_dict[passage_index]

    def transform_one_hot(
        self, categorical: List[int], passage_index: int, chunk_index: int = None
    ):
        key = passage_index
        if chunk_index is not None:
            key = (passage_index, chunk_index)
        if key not in self.cache_one_hot_dict.keys():
            return_list = []
            for el in categorical:
                if el in self.one_hot_dict.keys():
                    return_list.append(self.one_hot_dict[el])
                else:
                    return_list.append(None)
                    print(f"unable to encode onehot of NER/POS tag: {el}")
            self.cache_one_hot_dict[key] = return_list
        return self.cache_one_hot_dict[key]

    def get_categorical_in_dict(self, el):
        if el in self.categorical_dict.keys():
            return self.categorical_dict[el]
        else:
            print("element not found in categorical_encoder")
            return None

    def get_OHE_in_dict(self, el):
        if el in self.one_hot_dict.keys():
            return self.one_hot_dict[el]
        else:
            print("element not found in one_hot_encoder")
            return None

    def reset_cache(self):
        self.cache_categorical_dict = {}
        self.cache_one_hot_dict = {}

    def __init__(self, d: dict = None):
        if d is None:
            self.one_hot_dict = {}
            self.categorical_dict = {}
            self.cache_categorical_dict = {}
            self.cache_one_hot_dict = {}
        else:
            self.set_OHE_dicts(d)

    def get_OHE_dicts(self):
        return {
            "one_hot_dict": self.one_hot_dict,
            "categorical_dict": self.categorical_dict,
            "cache_categorical_dict": self.cache_categorical_dict,
            "cache_one_hot_dict": self.cache_one_hot_dict
        }

    def set_OHE_dicts(self, d: dict):
        self.one_hot_dict = d["one_hot_dict"]
        self.categorical_dict = d["categorical_dict"]
        self.cache_categorical_dict = d["cache_categorical_dict"]
        self.cache_one_hot_dict = d["cache_one_hot_dict"]
