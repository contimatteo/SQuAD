import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dropout, Add
from tensorflow.keras.optimizers import Adam, Optimizer

import utils.configs as Configs

from models.core import GloveEmbeddings, DrqaRnn, EnhancedProbabilities, WeightedSumCustom, WeightedSumSelfAttention
from models.core import AlignedAttention, BiLinearSimilarityAttention, BiLinearSimilarity
from models.core import drqa_crossentropy_loss, drqa_accuracy, Mask_layer
from utils import learning_rate

###

LOSS = [drqa_crossentropy_loss]

METRICS = [drqa_accuracy]

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

    def _build() -> Model:
        q_tokens = Input(shape=(N_Q_TOKENS, ))
        q_mask = Input(shape=(N_Q_TOKENS, ))
        p_mask = Input(shape=(N_P_TOKENS, ))
        p_tokens = Input(shape=(N_P_TOKENS, ))

        p_match = Input(shape=(N_P_TOKENS, DIM_EXACT_MATCH))
        p_pos = Input(shape=(N_P_TOKENS, N_POS_CLASSES))
        p_ner = Input(shape=(N_P_TOKENS, N_NER_CLASSES))
        p_tf = Input(shape=(N_P_TOKENS, DIM_TOKEN_TF))

        ### QUESTION ##############################################################

        ### embeddings
        q_embeddings = GloveEmbeddings(N_Q_TOKENS, embeddings_initializer)(q_tokens)

        ### lstm #
        q_rnn = DrqaRnn()(q_embeddings)
        # q_rnn = Mask_layer()(q_rnn, q_mask)

        ### self-attention (simplfied version)
        # q_encoding = WeightedSumSelfAttention()(q_rnn)
        # q_encoding = WeightedSum(q_rnn.shape[2], N_Q_TOKENS)(q_rnn)  ### --> (_,1,emb_dim)
        q_encoding = WeightedSumCustom(N_Q_TOKENS)(q_rnn)

        ### PASSAGE ###############################################################

        ### embeddings
        p_embeddings = GloveEmbeddings(N_P_TOKENS, embeddings_initializer)(p_tokens)

        ### aligend-attention
        p_attention = AlignedAttention()([p_embeddings, q_embeddings])
        # p_attention = Mask_layer()(p_attention, p_mask)

        ### lstm
        p_rnn = DrqaRnn()(
            Concatenate(axis=2)([p_attention, p_embeddings, p_match, p_pos, p_ner, p_tf])
        )

        ### OUTPUT ################################################################

        # ### similarity
        # out_probs = BiLinearSimilarityAttention()([p_rnn, q_encoding1])
        out_probs = BiLinearSimilarity()([p_rnn, p_mask, q_encoding])

        # out_probs = Mask_layer()(out_probs, p_mask)

        ### last bit
        out_probs = EnhancedProbabilities()(out_probs)

        # ###

        return Model(
            [q_tokens, q_mask, p_mask, p_tokens, p_match, p_pos, p_ner, p_tf],
            out_probs,
            name="DRQA"
        )

    #

    return _compile(_build())
