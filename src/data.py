import pandas as pd
from data.data import get_data
import numpy as np
import ast
from utils.data import get_argv
###


def test():
    my_str = "[[0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0] [0 0]]"
    my_str = my_str.replace("[ ", "[").replace(" ]", "]").replace(" ", ",")
    arr = ast.literal_eval(my_str)
    print(arr)
    arr = np.array(arr)
    print(arr)


def data_dataset():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    json_path = get_argv()
    df, df_np, glove_matrix, WTI = get_data(300, "data", debug=True, json_path=json_path)
    print("\n---------------\n")
    # print(df.columns)
    # print(df.shape)
    # print(df[0:4])
    print("\n---------------\n")
    question_index, passage, question, pos, ner, tf, exact_match = df_np
    print(f"INDEX\n{question_index[0:1]}\nPASSAGE\n{passage[0:1]}\nQUESTION\n{question[0:1]}\nPOS TAGGING\n{pos[0:1]}\nNAME ENTITY RECOGNITION\n{ner[0:1]}\nTERM FREQUENCY\n{tf[0:1]}\nEXACT MATCH\n{exact_match[0:1]}\n")

    print(
        f"INDEX\n{question_index[0:1].shape, question_index[0:1].dtype}\nPASSAGE\n{passage[0].shape, passage[0].dtype}\nQUESTION\n{question[0].shape, question[0].dtype}\nPOS TAGGING\n{pos[0].shape, pos[0].dtype}\nNAME ENTITY RECOGNITION\n{ner[0].shape, ner[0].dtype}\nTERM FREQUENCY\n{tf[0].shape, tf[0].dtype}\nEXACT MATCH\n{exact_match[0].shape, exact_match[0].dtype}\n"
    )

    print(f'\nGLOVE MATRIX\n{glove_matrix.shape, glove_matrix.dtype}')


def data_eval():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    json_path = get_argv()
    df, df_np, glove_matrix, WTI = get_data(300, "original", debug=True, json_path=json_path)
    print("\n---------------\n")
    # print(df.columns)
    # print(df.shape)
    # print(df[0:4])
    print("\n---------------\n")
    eval_index, eval_passage = df_np
    print(f"\nEVALUATION_INDEX\n{eval_index[0:1]}\nEVALUATION_PASSAGE\n{eval_passage[0:1]}")

    print(f'\nGLOVE MATRIX\n{glove_matrix.shape, glove_matrix.dtype}')


def data_label():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    json_path = get_argv()
    df, df_np, glove_matrix, WTI = get_data(300, "label", debug=True, json_path=json_path)
    print("\n---------------\n")
    # print(df.columns)
    # print(df.shape)
    # print(df[0:4])
    print("\n---------------\n")
    label = df_np
    if label is not None:
        print(f"LABEL{label[0:1]}\n")
        print(f"LABEL{label[0].shape, label[0].dtype}\n")

    print(f'\nGLOVE MATRIX\n{glove_matrix.shape, glove_matrix.dtype}')


def data():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    json_path = get_argv()
    df, df_np, glove_matrix, WTI = get_data(300, debug=True, json_path=json_path)
    print("\n---------------\n")
    # print(df.columns)
    # print(df.shape)
    # print(df[0:4])
    print("\n---------------\n")
    question_index, passage, question, pos, ner, tf, exact_match, label, eval_index, eval_passage = df_np
    print(f"INDEX\n{question_index[0:1]}\nPASSAGE\n{passage[0:1]}\nQUESTION\n{question[0:1]}\nPOS TAGGING\n{pos[0:1]}\nNAME ENTITY RECOGNITION\n{ner[0:1]}\nTERM FREQUENCY\n{tf[0:1]}\nEXACT MATCH\n{exact_match[0:1]}\n")
    if label is not None:
        print(f"LABEL{label[0:1]}\n")
    print(
        f"INDEX\n{question_index[0:1].shape, question_index[0:1].dtype}\nPASSAGE\n{passage[0].shape, passage[0].dtype}\nQUESTION\n{question[0].shape, question[0].dtype}\nPOS TAGGING\n{pos[0].shape, pos[0].dtype}\nNAME ENTITY RECOGNITION\n{ner[0].shape, ner[0].dtype}\nTERM FREQUENCY\n{tf[0].shape, tf[0].dtype}\nEXACT MATCH\n{exact_match[0].shape, exact_match[0].dtype}\n"
    )
    if label is not None:
        print(f"LABEL{label[0].shape, label[0].dtype}\n")

    print(f"\nEVALUATION_INDEX\n{eval_index[0:1]}\nEVALUATION_PASSAGE\n{eval_passage[0:1]}")

    print("ROW_NUMBER: " + str(question_index.shape[0]))

    print(f'\nGLOVE MATRIX\n{glove_matrix.shape, glove_matrix.dtype}')


###

if __name__ == "__main__":
    data_dataset()
