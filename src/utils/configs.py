import os

###########
##  APP  ##

APP_DEBUG: bool = True

WANDB_DISABLED = "WANDB_DISABLED" not in os.environ or os.environ["WANDB_DISABLED"] == "true"

###################
##  NN FEATURES  ##

N_QUESTION_TOKENS: int = 30
N_PASSAGE_TOKENS: int = 50

N_NER_CLASSES: int = 13
N_POS_CLASSES: int = 46

DIM_EMBEDDING: int = 300
DIM_EXACT_MATCH: int = 3
DIM_TOKEN_TF: int = 1

##################
##  NN + TRAIN  ##

NN_EPOCHS = 3
NN_BATCH_SIZE = 8
NN_LEARNING_RATE_TYPE = "static"

N_KFOLD_BUCKETS = 5
