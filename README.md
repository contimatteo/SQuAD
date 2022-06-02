# SQuAD-PMFL
NLP system trained on Stanford Question Answering Dataset (SQuAD). SQuAD tests the ability of a system to not only answer reading comprehension questions, but also abstain when presented with a question that cannot be answered based on the provided paragraph.



Prepare train data

$python train.py



Test data

$python test.py test_set.json



Evaluation

$python evaluate.py test_set.json answers.training.pred.json

