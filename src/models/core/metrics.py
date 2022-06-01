from tensorflow.keras.metrics import CategoricalAccuracy

import utils.configs as Configs

###


def __drqa_accuracy(y_true, y_pred):
    m = CategoricalAccuracy()
    m.update_state(y_true, y_pred)
    return m.result()


def __drqa_accuracy_without_complementary_bit(y_true, y_pred):
    return __drqa_accuracy(y_true[:, :-1], y_pred[:, :-1])


#


def __drqa_accuracy_start(y_true, y_pred):
    return __drqa_accuracy_without_complementary_bit(y_true[:, :, 0], y_pred[:, :, 0])


def __drqa_accuracy_end(y_true, y_pred):
    return __drqa_accuracy_without_complementary_bit(y_true[:, :, 1], y_pred[:, :, 1])


#


class DrQAMetrics:

    @staticmethod
    def accuracy(y_true, y_pred):

        def _aggregate(s_acc, e_acc):
            return (s_acc + e_acc) / 2

        s_acc = __drqa_accuracy_start(y_true, y_pred)
        e_acc = __drqa_accuracy_end(y_true, y_pred)

        return _aggregate(s_acc, e_acc)
