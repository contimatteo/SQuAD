import os
from dotenv import load_dotenv

load_dotenv()

###########
##  APP  ##

APP_DEBUG: bool = True

WANDB_DISABLED = "WANDB_DISABLED" not in os.environ or os.environ["WANDB_DISABLED"] == "true"

CUDA_ENABLED = "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "-1"

COMPLEMENTAR_BIT = True

###################
##  NN FEATURES  ##

N_QUESTION_TOKENS: int = 30
N_PASSAGE_TOKENS: int = 50

N_NER_CLASSES: int = 13
N_POS_CLASSES: int = 46

DIM_EMBEDDING: int = 50
DIM_EXACT_MATCH: int = 3
DIM_TOKEN_TF: int = 1

##################
##  NN + TRAIN  ##

NN_EPOCHS = 50  # 50    inizia ad andar bene a 30
NN_BATCH_SIZE = 64  # 16
NN_LEARNING_RATE = 5e-3
NN_LEARNING_RATE_TYPE = "static"

N_KFOLD_BUCKETS = 5
