from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate, Flatten
from tensorflow.keras.optimizers import Adam

from models.layers import EmbeddingLayers, DenseLayers, RnnLayers

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
    xqi = Input(shape=(5, ))
    xpi = Input(shape=(10, ))

    # Question
    xqe = EmbeddingLayers.glove(5)(xqi)
    xqr = RnnLayers.drqa_question()(xqe)
    xqd = DenseLayers.regularized()(xqr)

    # Passage
    xpe = EmbeddingLayers.glove(10)(xpi)
    xpr = RnnLayers.drqa_passage()(xpe)
    xpd = DenseLayers.regularized()(xpr)

    # Output
    xqo = Flatten()(xqd)
    xpo = Flatten()(xpd)
    xo = Concatenate()([xqo, xpo])

    #

    model = Model([xqd, xpd], xo)

    return _compile(model)
