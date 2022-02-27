from typing import Any

from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
from tensorflow.keras.callbacks import LearningRateScheduler
#  from tensorflow_addons.optimizers import CyclicalLearningRate
import math

import utils.configs as Configs

###


def __lr_cosine_decay(initial_lr=1e-3, decay_steps=Configs.NN_EPOCHS, alpha=0.05) -> Any:
    return CosineDecay(initial_lr, decay_steps, alpha=alpha)


def __lr_exponential_decay(initial_lr=1e-3, decay_steps=Configs.NN_EPOCHS, decay_rate=0.96) -> Any:
    return ExponentialDecay(initial_lr, decay_steps=decay_steps, decay_rate=decay_rate)


# def __cyclic_decay() -> Any:
#     LEARNING_RATE_MIN_VALUE = Configs.LEARNING_RATE / 10
#     step_size = 2 * Configs.BATCH_SIZE
#     scale_fn = lambda x: 1 / (2.**(x - 1))
#     return CyclicalLearningRate(
#         LEARNING_RATE_MIN_VALUE, Configs.LEARNING_RATE, step_size=step_size, scale_fn=scale_fn
#     )

###


def scheduler_static(epoch, lr):
    return Configs.NN_LEARNING_RATE


def scheduler_custom(epoch, *_):
    # epoch: 0..Configs.NN_EPOCHS - 1

    lr_warmup = Configs.NN_WARMUP_LEARNING_RATE
    lr0 = Configs.NN_START_LEARNING_RATE
    lrE = Configs.NN_END_LEARNING_RATE
    epoch_warmup = Configs.NN_EPOCHS_WARMUP
    epoch_decay = Configs.NN_EPOCHS_DECAY

    # epoch_static = Configs.NN_EPOCHS_STATIC

    def cosine_decay_fn(e, E):
        lr = lrE + (0.5 * (lr0 - lrE) * (1 + math.cos((math.pi * e) / E)))
        return lr

    if epoch < epoch_warmup:
        return lr_warmup
    if epoch < epoch_warmup + epoch_decay:
        return cosine_decay_fn(epoch - epoch_warmup, epoch_decay)

    return lrE


def scheduler_cyclic_cosine_decay(epoch, *_):
    # epoch: 0..Configs.NN_EPOCHS - 1

    lr0 = Configs.NN_START_LEARNING_RATE
    lrE = Configs.NN_END_LEARNING_RATE
    cycle = Configs.NN_EPOCHS_CYCLE

    def cosine_decay_fn(e, E):
        lr = lrE + (0.5 * (lr0 - lrE) * (1 + math.cos((math.pi * e) / E)))
        return lr

    return cosine_decay_fn(epoch % cycle, cycle)


def learning_rate(config_name: str) -> Any:

    if config_name == "cosine":
        return __lr_cosine_decay()

    if config_name == "exponential":
        return __lr_exponential_decay()

    if config_name == "custom":
        # return LearningRateScheduler(scheduler_custom)
        return LearningRateScheduler(scheduler_cyclic_cosine_decay)

    if config_name == "static":
        return LearningRateScheduler(scheduler_static)

    raise Exception("[learning_rate]: invalid  `config_name` parameter.")
