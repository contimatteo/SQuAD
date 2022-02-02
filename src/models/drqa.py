import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam, Optimizer

import utils.configs as Configs

from models.core import GloveEmbeddings, DrqaRnn
from models.core import WeightedSumSelfAttention, AlignedAttention, BiLinearSimilarityAttention
from models.core import drqa_crossentropy
# from models.core import drqa_accuracy, drqa_accuracy_start, drqa_accuracy_end

###

LEARNING_RATE = 1e-4

LOSS = [drqa_crossentropy]
METRICS = ['categorical_accuracy']  # drqa_accuracy, drqa_accuracy_start, drqa_accuracy_end]

###


def _optimizer() -> Optimizer:
    return Adam(learning_rate=LEARNING_RATE)


def _compile(model) -> Model:
    model.compile(loss=LOSS, optimizer=_optimizer(), metrics=METRICS)
    return model


###


# pylint: disable=invalid-name
def DRQA(embeddings_initializer: np.ndarray) -> Model:
    N_Q_TOKENS = Configs.N_QUESTION_TOKENS
    N_P_TOKENS = Configs.N_PASSAGE_TOKENS
    DIM_EXACT_MATCH = Configs.DIM_EXACT_MATCH
    N_POS_CLASSES = Configs.N_POS_CLASSES
    N_NER_CLASSES = Configs.N_NER_CLASSES
    DIM_TOKEN_TF = Configs.DIM_TOKEN_TF

    def _build() -> Model:
        q_tokens = Input(shape=(N_Q_TOKENS, ))
        p_tokens = Input(shape=(N_P_TOKENS, ))

        p_match = Input(shape=(N_P_TOKENS, DIM_EXACT_MATCH))
        p_pos = Input(shape=(N_P_TOKENS, N_POS_CLASSES))
        p_ner = Input(shape=(N_P_TOKENS, N_NER_CLASSES))
        p_tf = Input(shape=(N_P_TOKENS, DIM_TOKEN_TF))

        ### QUESTION ##############################################################

        ### embeddings
        q_embeddings = GloveEmbeddings(N_Q_TOKENS, embeddings_initializer)(q_tokens)

        ### lstm
        q_rnn = DrqaRnn()(q_embeddings)

        ### self-attention (simplfied version)
        q_encoding = WeightedSumSelfAttention()(q_rnn)

        ### PASSAGE ###############################################################

        ### embeddings
        p_embeddings = GloveEmbeddings(N_P_TOKENS, embeddings_initializer)(p_tokens)

        ### aligend-attention
        p_attention = AlignedAttention()([p_embeddings, q_embeddings])

        ### lstm
        p_rnn = DrqaRnn()(
            Concatenate(axis=2)([p_attention, p_embeddings, p_match, p_pos, p_ner, p_tf])
        )

        ### OUTPUT ################################################################

        ### similarity
        out = BiLinearSimilarityAttention()([p_rnn, q_encoding])

        return Model([q_tokens, p_tokens, p_match, p_pos, p_ner, p_tf], out)

    #

    return _compile(_build())
