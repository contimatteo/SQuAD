import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dropout, Conv1D
from tensorflow.keras.optimizers import Adam, Optimizer

import utils.configs as Configs

from models.core import GloveEmbeddings, DrqaRnn, EnhancedProbabilities
from models.core import WeightedSumSelfAttention, AlignedAttention, BiLinearSimilarityAttention, BiLinearSimilarity
from models.core import drqa_crossentropy_loss, drqa_accuracy_metric, drqa_prob_sum_loss, drqa_start_accuracy_metric, drqa_end_accuracy_metric
from utils import learning_rate

###

# LOSS = ['categorical_crossentropy']
LOSS = [drqa_crossentropy_loss]

METRICS = [drqa_start_accuracy_metric, drqa_end_accuracy_metric, drqa_crossentropy_loss]

###


def _optimizer() -> Optimizer:
    lr = learning_rate("static")
    return Adam(learning_rate=lr, clipnorm=1)


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
    EMBEDDING_DIM = Configs.DIM_EMBEDDING

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
        # q_embeddings = Dropout(.3)(q_embeddings)

        ### lstm
        q_rnn = DrqaRnn()(q_embeddings)

        ### self-attention (simplfied version)
        q_encoding = WeightedSumSelfAttention()(q_rnn)
        q_encoding1 = Conv1D(q_rnn.shape[2], N_Q_TOKENS)(q_rnn)  ### --> (_,1,emb_dim)
        # print()
        # print("q_rnn: ", q_rnn.shape)
        # print("q_encoding: ", q_encoding.shape)
        # print("q_encoding1: ", q_encoding1.shape)
        # print()
        # print()

        ### PASSAGE ###############################################################

        ### embeddings
        p_embeddings = GloveEmbeddings(N_P_TOKENS, embeddings_initializer)(p_tokens)
        # p_embeddings = Dropout(.3)(p_embeddings)

        ### aligend-attention
        p_attention = AlignedAttention()([p_embeddings, q_embeddings])

        ### lstm
        p_rnn = DrqaRnn()(
            Concatenate(axis=2)([p_attention, p_embeddings, p_match, p_pos, p_ner, p_tf])
        )

        # print()
        # print("p_rnn: ", p_rnn.shape)
        # print()
        # print()
        # raise Exception("stop")
        ### OUTPUT ################################################################

        # ### similarity
        # out_probs = BiLinearSimilarityAttention()([p_rnn, q_encoding])

        # out_probs = BiLinearSimilarity()([p_rnn, q_encoding])
        out_probs = BiLinearSimilarity()([p_rnn, q_encoding1])

        # ### last bit
        # out_probs = EnhancedProbabilities()(out_probs)

        # ###

        return Model([q_tokens, p_tokens, p_match, p_pos, p_ner, p_tf], out_probs, name="DRQA")

    #

    return _compile(_build())
