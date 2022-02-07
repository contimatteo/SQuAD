import os
from utils.data import get_project_directory


def evaluate(data_path: str, prediction_path: str, output_file=None, na_prob_path=None):
    command = f'{os.path.join("utils", "evaluate.py")} "{str(data_path)}" "{str(prediction_path)}"'
    if output_file is not None:
        command = command + ' -o "' + output_file + '"'
    if na_prob_path is not None:
        command = command + ' -n "' + na_prob_path + '"'
    print(command)
    os.system(command)


###

if __name__ == "__main__":
    eval_folder = os.path.join(os.path.join(get_project_directory(), "data"), "evaluation")
    print(eval_folder)
    evaluate(os.path.join(eval_folder, "data.json"), os.path.join(eval_folder, "preds.json"), output_file=os.path.join(eval_folder, "eval.json"))
