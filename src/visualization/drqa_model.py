from tensorflow.keras.utils import plot_model

from models import DRQA

###


def plot_drqa_model(glove_matrix):
    model = DRQA(glove_matrix)

    plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
