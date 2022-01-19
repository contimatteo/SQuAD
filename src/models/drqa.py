from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate, Flatten, Dense
from tensorflow.keras.optimizers import Adam, Adamax

import utils.configs as Configs
from models.layers import EmbeddingLayers, DenseLayers, RnnLayers, AttentionLayers

###


def _optimizer():
    lr = 1e-3
    return Adam(learning_rate=lr)


def _compile(model):
    loss = ['mae']
    metrics = ['accuracy']

    model.compile(loss=loss, optimizer=_optimizer(), metrics=metrics)

    return model


def DrQA() -> Model:
    n_q_tokens = Configs.N_QUESTION_TOKENS
    n_p_tokens = Configs.N_PASSAGE_TOKENS

    xqi = Input(shape=(n_q_tokens, ))
    xpi = Input(shape=(n_p_tokens, ))

    print()
    print()

    # Question
    xq = EmbeddingLayers.glove(n_q_tokens)(xqi)
    xq = RnnLayers.drqa()(xq)
    xq = AttentionLayers.weighted_sum()(xq)

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
