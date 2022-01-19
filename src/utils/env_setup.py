import os
import logging
import warnings
import wandb

from dotenv import load_dotenv

###

load_dotenv()

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

###

wandb.init()
