# pylint: disable=wrong-import-order,wrong-import-position
from dotenv import load_dotenv

load_dotenv()

###

import os
import logging
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

###

import wandb

if not "WANDB_DISABLED" in os.environ or os.environ["WANDB_DISABLED"] == "true":
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
else:
    os.environ["WANDB_DISABLED"] = "false"
    wandb.init()

###

import tensorflow as tf

tf.config.run_functions_eagerly(True)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         pass
