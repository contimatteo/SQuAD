###########
##  APP  ##

APP_DEBUG: bool = True

###################
##  NN FEATURES  ##

N_QUESTION_TOKENS: int = 30
N_PASSAGE_TOKENS: int = 50  ### TODO: set the right value ...

N_NER_CLASSES: int = 13
N_POS_CLASSES: int = 46

DIM_EMBEDDING: int = 300
DIM_EXACT_MATCH: int = 3
DIM_TOKEN_TF: int = 1

NN_EPOCHS = 20
NN_BATCH_SIZE = 256
NN_LEARNING_RATE = 1e-3
NN_LEARNING_RATE_TYPE = "static"
