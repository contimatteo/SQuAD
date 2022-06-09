# SQuAD-PMFL
NLP system trained on Stanford Question Answering Dataset (SQuAD). SQuAD tests the ability of a system to not only answer reading comprehension questions, but also abstain when presented with a question that cannot be answered based on the provided paragraph.


## Prepare train data

$python src/train.py


## Test data

$python src/compute_answers.py *path_to_json_file*


## Evaluation

$python src/evaluate.py *path_to_json_file* data/predictions/answers.pred.json

