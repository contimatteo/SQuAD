from typing import Any, AnyStr
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
# from tensorflow_addons.optimizers import CyclicalLearningRate
from wandb import Config

import utils.configs as Configs

from models.core import GloveEmbeddings, DrqaRnn, EnhancedProbabilities
from models.core import WeightedSumSelfAttention, AlignedAttention, BiLinearSimilarityAttention
from models.core import drqa_crossentropy
from models.core import start_accuracy, end_accuracy, tot_accuracy

###

LOSS = ['binary_crossentropy']  # [drqa_crossentropy]
# METRICS = [start_accuracy, end_accuracy, tot_accuracy, drqa_crossentropy]  # 'categorical_accuracy'
METRICS = [tot_accuracy, drqa_crossentropy]

###


def _adaptive_learning_rate(name="cosine") -> Any:

    def __cosine_decay() -> Any:
        initial_lr = Configs.LEARNING_RATE
        decay_steps = Configs.EPOCHS
        alpha = 0.1

        return CosineDecay(initial_lr, decay_steps, alpha=alpha)

    def __exponential_decay() -> Any:
        initial_lr = Configs.LEARNING_RATE
        decay_steps = Configs.EPOCHS
        decay_rate = 0.96

        return ExponentialDecay(initial_lr, decay_steps=decay_steps, decay_rate=decay_rate)

    # def __cyclic_decay() -> Any:
    #     LEARNING_RATE_MIN_VALUE = Configs.LEARNING_RATE / 10
    #     step_size = 2 * Configs.BATCH_SIZE
    #     scale_fn = lambda x: 1 / (2.**(x - 1))

    #     return CyclicalLearningRate(
    #         LEARNING_RATE_MIN_VALUE,
    #         Configs.LEARNING_RATE,
    #         step_size=step_size,
    #         scale_fn=scale_fn
    #     )

    if name == "cosine":
        return __cosine_decay()
    if name == "exponential":
        return __exponential_decay()
    # if name == "cyclic":
    #     return __cyclic_decay()

    raise Exception("[_adaptive_learning_rate]: invalid  `name` parameter.")


###


def _optimizer() -> Optimizer:
    # lr = Configs.LEARNING_RATE
    lr = _adaptive_learning_rate(name="exponential")

    return Adam(learning_rate=lr)


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
        out_probabilities = BiLinearSimilarityAttention()([p_rnn, q_encoding])

        ### last bit
        out_probabilities = EnhancedProbabilities()(out_probabilities)

        ###

        return Model([q_tokens, p_tokens, p_match, p_pos, p_ner, p_tf], out_probabilities)

    #

    return _compile(_build())
