from typing import List
import numpy as np


class OneHotEncoder:
    one_hot_dict = {}

    def fit(self, pos_list):
        pos_len = len(pos_list)
        id_m = np.identity(pos_len, dtype="int")
        pos_dict = {}
        for i, el in enumerate(pos_list):
            self.one_hot_dict[el] = id_m[i]
        return self.one_hot_dict

    def get_one_hot_dict(self):
        return self.one_hot_dict

    def transform(self, df_row: List[str]):
        return_list = []
        for el in df_row:
            if el in self.one_hot_dict.keys():
                return_list.append(self.one_hot_dict[el])
            else:
                return_list.append(None)
        return return_list

