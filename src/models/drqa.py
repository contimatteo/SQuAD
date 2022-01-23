from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate, Flatten, Dense
from tensorflow.keras.optimizers import Adam

import utils.configs as Configs
from models.layers import EmbeddingLayers, RnnLayers, AttentionLayers

###


def _optimizer():
    return Adam(learning_rate=1e-4)


def _compile(model):
    model.compile(loss=['mse'], optimizer=_optimizer(), metrics=['mse'])
    return model


###


def DRQA() -> Model:
    n_q_tokens = Configs.N_QUESTION_TOKENS
    n_p_tokens = Configs.N_PASSAGE_TOKENS

    xqi = Input(shape=(n_q_tokens, ))
    xpi = Input(shape=(n_p_tokens, ))

    # Question
    xq = EmbeddingLayers.glove(n_q_tokens)(xqi)
    xq = RnnLayers.drqa()(xq)
    xq = AttentionLayers.question_encoding()(xq)

    # Passage
    xp = EmbeddingLayers.glove(n_p_tokens)(xpi)
    xp = RnnLayers.drqa()(xp)

    # Output
    xq = Flatten()(xq)
    xp = Flatten()(xp)
    xo = Concatenate()([xq, xp])
    xo = Dense(1)(xo)

    #

    model = Model([xqi, xpi], xo)

    return _compile(model)
