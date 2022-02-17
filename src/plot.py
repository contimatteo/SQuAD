# pylint: disable=unused-import
import numpy as np

import utils.env_setup
import utils.configs as Configs

from visualization import plot_drqa_model

###

glove_matrix = np.zeros((1, Configs.DIM_EMBEDDING))
p_mask = np.zeros((50,))
q_mask = np.zeros((30,))
###

if __name__ == "__main__":
    plot_drqa_model(glove_matrix, p_mask, q_mask)
