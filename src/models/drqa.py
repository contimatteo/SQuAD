from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Attention, Dense
from tensorflow.keras.layers import Concatenate, Flatten
from tensorflow.keras.optimizers import Adam

import utils.configs as Configs
from models.layers import EmbeddingLayers, RnnLayers, AttentionLayers

###


def _optimizer():
    return Adam(learning_rate=1e-4)


def _compile(model):
    model.compile(loss=['mse'], optimizer=_optimizer(), metrics=['mse'], run_eagerly=True)
    return model


###


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
    q_att = AttentionLayers.question_encoding()(q_rnn)

    ### PASSAGE ###############################################################

    ### embeddings
    p_embd = EmbeddingLayers.glove(n_p_tokens)(p_xi)

    ### aligend-attention
    # p_att = Attention()([p_embd, q_embd])  # ([query, keys/values])
    # p_att = AttentionLayers.passage_embeddings()([p_embd, q_embd])  # ([query, keys/values])
    p_att = AttentionLayers.alignment()(p_embd, q_embd)

    ### lstm (features)
    p_concat = Concatenate(axis=2)([p_embd, p_att])
    ### lstm
    p_rnn = RnnLayers.drqa()(p_concat)

    ### OUTPUT ################################################################

    q_out = Flatten()(q_att)
    p_out = Flatten()(p_rnn)
    out = Concatenate()([q_out, p_out])
    out = Dense(1)(out)

    #

    model = Model([q_xi, p_xi], out)

    return _compile(model)
