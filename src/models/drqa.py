from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate, Flatten
from tensorflow.keras.optimizers import Adam, Optimizer

import utils.configs as Configs

from models.core import EmbeddingLayers, RnnLayers, AttentionLayers
from models.core import drqa_categorical_crossentropy

###

LEARNING_RATE = 1e-3

LOSS = [drqa_categorical_crossentropy]  # ['binary_crossentropy']
METRICS = ['categorical_accuracy']

###


def _optimizer() -> Optimizer:
    return Adam(learning_rate=LEARNING_RATE)


def _compile(model) -> Model:
    model.compile(loss=LOSS, optimizer=_optimizer(), metrics=METRICS)  # run_eagerly=True
    return model


###


# pylint: disable=invalid-name
def DRQA() -> Model:
    n_q_tokens = Configs.N_QUESTION_TOKENS
    n_p_tokens = Configs.N_PASSAGE_TOKENS

    def _build() -> Model:

        q_xi = Input(shape=(n_q_tokens, ))
        p_xi = Input(shape=(n_p_tokens, ))

        ### QUESTION ##############################################################

        ### embeddings
        q_embd = EmbeddingLayers.glove(n_q_tokens)(q_xi)

        ### lstm
        q_rnn = RnnLayers.drqa()(q_embd)

        ### self-attention (simplfied version)
        q_enc = AttentionLayers.question_encoding()(q_rnn)

        ### PASSAGE ###############################################################

        ### embeddings
        p_embd = EmbeddingLayers.glove(n_p_tokens)(p_xi)

        ### aligend-attention
        p_att = AttentionLayers.alignment()([p_embd, q_embd])

        ### lstm (features)
        p_embd_att = Concatenate(axis=2)([p_embd, p_att])
        ### lstm
        p_rnn = RnnLayers.drqa()(p_embd_att)

        ### OUTPUT ################################################################

        ### similarity
        out = AttentionLayers.bilinear_similarity()([p_rnn, q_enc])

        return Model([q_xi, p_xi], out)

    return _compile(_build())
