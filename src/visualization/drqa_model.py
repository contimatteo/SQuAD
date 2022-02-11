from tensorflow.keras.utils import plot_model

from models import DRQA

###


def plot_drqa_model(glove_matrix):
    model = DRQA(glove_matrix)

    plot_model(model, "tmp/network.png", show_shapes=True)
