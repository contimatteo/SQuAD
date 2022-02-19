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

if not "WANDB_DISABLED" in os.environ or os.environ["WANDB_DISABLED"] == "true":
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"
else:
    import wandb
    os.environ["WANDB_DISABLED"] = "false"
    wandb.init()

###

import tensorflow as tf

### use 'eager' execution
tf.config.run_functions_eagerly(True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         pass
