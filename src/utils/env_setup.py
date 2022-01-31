###

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
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_MODE"] = "offline"

wandb.init()

###

import tensorflow as tf
#
# tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
