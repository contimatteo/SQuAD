from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional, LSTM

###


class EmbeddingLayers():

    @staticmethod
    def glove(input_length):
        assert isinstance(input_length, int)

        input_dim = 100  # size of the vocabulary
        output_dim = 50  # dimension of the dense embedding

        return Embedding(input_dim, output_dim, input_length=input_length, trainable=True)


###


class DenseLayers():

    @staticmethod
    def regularized():
        return Dense(5)


###


class RnnLayers():

    @staticmethod
    def drqa_question():
        forward = LSTM(5, go_backwards=False, return_sequences=True)
        backward = LSTM(5, go_backwards=True, return_sequences=True)

        return Bidirectional(forward, backward_layer=backward)

    @staticmethod
    def drqa_passage():
        forward = LSTM(5, go_backwards=False, return_sequences=True)
        backward = LSTM(5, go_backwards=True, return_sequences=True)

        return Bidirectional(forward, backward_layer=backward)
