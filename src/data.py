import pandas as pd
from data.data import get_data, load_data
import numpy as np
import ast
from utils.data import get_argv
###


def load():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    json_path = get_argv()
    load_data(300, debug=True, json_path=json_path)


def data():
    load()
    print("\n---------------\n")
    data_data = get_data("data")
    question_index, passage, question, pos, ner, tf, exact_match = data_data
    # print(f"INDEX\n{question_index[0:1]}\nPASSAGE\n{passage[0:1]}\nQUESTION\n{question[0:1]}\nPOS TAGGING\n{pos[0:1]}\nNAME ENTITY RECOGNITION\n{ner[0:1]}\nTERM FREQUENCY\n{tf[0:1]}\nEXACT MATCH\n{exact_match[0:1]}\n")
    print(
        f"INDEX\n{question_index[0:1].shape, question_index[0:1].dtype}\nPASSAGE\n{passage[0].shape, passage[0].dtype}\nQUESTION\n{question[0].shape, question[0].dtype}\nPOS TAGGING\n{pos[0].shape, pos[0].dtype}\nNAME ENTITY RECOGNITION\n{ner[0].shape, ner[0].dtype}\nTERM FREQUENCY\n{tf[0].shape, tf[0].dtype}\nEXACT MATCH\n{exact_match[0].shape, exact_match[0].dtype}\n"
    )
    print("\n---------------\n")
    data_data = get_data("label")
    label = data_data
    # if label is not None:
    #     print(f"LABEL{label[0:1]}\n")
    # print(
    #     f"INDEX\n{question_index[0:1].shape, question_index[0:1].dtype}\nPASSAGE\n{passage[0].shape, passage[0].dtype}\nQUESTION\n{question[0].shape, question[0].dtype}\nPOS TAGGING\n{pos[0].shape, pos[0].dtype}\nNAME ENTITY RECOGNITION\n{ner[0].shape, ner[0].dtype}\nTERM FREQUENCY\n{tf[0].shape, tf[0].dtype}\nEXACT MATCH\n{exact_match[0].shape, exact_match[0].dtype}\n"
    # )
    if label is not None:
        print(f"LABEL{label[0].shape, label[0].dtype}\n")

    print("\n---------------\n")
    data_data = get_data("original")
    eval_index, eval_passage, question_passage = data_data
    print(f"\nEVALUATION_INDEX\n{eval_index[0:1]}\nEVALUATION_PASSAGE\n{eval_passage[0:1]}\nEVALUATION_QUESTION\n{question_passage[0:1]}")

    print("\n---------------\n")
    data_data = get_data("glove")
    glove_matrix = data_data
    print(f'\nGLOVE MATRIX\n{glove_matrix.shape, glove_matrix.dtype}')

    # print("ROW_NUMBER: " + str(question_index.shape[0]))


###

if __name__ == "__main__":
    data()
