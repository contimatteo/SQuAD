from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate, Flatten
from tensorflow.keras.optimizers import Adam, Optimizer

import utils.configs as Configs
from models.layers import EmbeddingLayers, RnnLayers, AttentionLayers, Customlayers

###

learning_rate = 1e-4

loss = ['binary_crossentropy']
metrics = ['binary_accuracy']


def _optimizer() -> Optimizer:
    return Adam(learning_rate=learning_rate)


def _compile(model) -> None:
    model.compile(loss=loss, optimizer=_optimizer(), metrics=metrics)  # run_eagerly=True


###


# pylint: disable=invalid-name
def DRQA() -> Model:
    n_q_tokens = Configs.N_QUESTION_TOKENS
    n_p_tokens = Configs.N_PASSAGE_TOKENS

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
    out = Customlayers.embeddings_similarity()([p_rnn, q_enc])
    out = Flatten()(out)

    # q_out = Flatten()(q_enc)
    # p_out = Flatten()(p_rnn)
    # out = Concatenate()([q_out, p_out])
    # out = Dense(1)(out)

    ### COMPILE ###############################################################

    model = Model([q_xi, p_xi], out)

    _compile(model)

    return model
