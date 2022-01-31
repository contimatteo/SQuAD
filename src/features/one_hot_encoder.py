from typing import List
import numpy as np


class OneHotEncoder:
    # one_hot_dict = {}
    # cache_dict = {}

    def fit(self, pos_list):
        # self.one_hot_dict = {}
        pos_len = len(pos_list)
        # print(pos_list)
        id_m = np.identity(pos_len, dtype="int")
        pos_dict = {}
        for i, el in enumerate(pos_list):
            self.one_hot_dict[el] = id_m[i]
        return self.one_hot_dict

    def get_one_hot_dict(self):
        return self.one_hot_dict

    def transform(self, df_row: List[str], passage_index: int):
        if passage_index not in self.cache_dict.keys():
            return_list = []
            for el in df_row:
                if el in self.one_hot_dict.keys():
                    return_list.append(self.one_hot_dict[el])
                else:
                    return_list.append(None)
                    print(f"unable to encode onehot of NER/POS tag: {el}")
            self.cache_dict[passage_index] = return_list
        return self.cache_dict[passage_index]

    def get_OHE_in_dict(self, el):
        if el in self.one_hot_dict.keys():
            return self.one_hot_dict[el]
        else:
            print("element not found in one_hot_encoder")
            return None

    def __init__(self):
        self.one_hot_dict = {}
        self.cache_dict = {}


