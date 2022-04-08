from tensorflow import convert_to_tensor
from tensorflow.keras.metrics import CategoricalAccuracy

import utils.configs as Configs

###


def __drqa_accuracy(y_true, y_pred):
    m = CategoricalAccuracy()
    m.update_state(y_true, y_pred)
    return m.result()


def __drqa_accuracy_without_c_bit(y_true, y_pred):
    return __drqa_accuracy(y_true[:, :-1], y_pred[:, :-1])


def __drqa_accuracy_filtered_exact_c_bit(y_true, y_pred):

    def filter_data():
        y_true_new = convert_to_tensor(
            [y_true[i, :] for i in range(y_true.shape[0]) if y_true[i, -1] != 1]
        )
        y_pred_new = convert_to_tensor(
            [y_pred[i, :] for i in range(y_pred.shape[0]) if y_pred[i, -1] != 1]
        )
        return y_true_new, y_pred_new

    y_true, y_pred = filter_data()
    if y_pred.shape[0] == 0:
        return 0

    return __drqa_accuracy(y_true, y_pred)


# def __drqa_accuracy_filtered(y_true, y_pred, accuracy_type="answer"):
#     def filter_data():
#         if accuracy_type == "answer":
#             y_true_index_filtered = [i for i in range(y_true.shape[0]) if y_true[i, -1] != 1]
#         else:
#             y_true_index_filtered = [i for i in range(y_true.shape[0]) if y_true[i, -1] == 1]
#         y_true_new = convert_to_tensor([y_true[i, :] for i in y_true_index_filtered])
#         y_pred_new = convert_to_tensor([y_pred[i, :] for i in y_true_index_filtered])
#         return y_true_new, y_pred_new
#     y_true, y_pred = filter_data()
#     if y_true.shape[0] == 0:
#         return 0
#     return __drqa_accuracy(y_true, y_pred)
#
# def drqa_start_accuracy_metric_filtered_exact_c_bit(y_true, y_pred):
#     return __drqa_accuracy_filtered_exact_c_bit(y_true[:, :, 0], y_pred[:, :, 0])
#
# def drqa_end_accuracy_metric_filtered_exact_c_bit(y_true, y_pred):
#     return __drqa_accuracy_filtered_exact_c_bit(y_true[:, :, 1], y_pred[:, :, 1])

#


def drqa_accuracy_start(y_true, y_pred, accuracy_type="answer"):
    # return __drqa_accuracy_filtered(y_true[:, :, 0], y_pred[:, :, 0], accuracy_type)
    return __drqa_accuracy_without_c_bit(y_true[:, :, 0], y_pred[:, :, 0])


def drqa_accuracy_end(y_true, y_pred, accuracy_type="answer"):
    # return __drqa_accuracy_filtered(y_true[:, :, 1], y_pred[:, :, 1], accuracy_type)
    return __drqa_accuracy_without_c_bit(y_true[:, :, 1], y_pred[:, :, 1])


#


def drqa_accuracy(y_true, y_pred):

    def _aggregate(s_acc, e_acc):
        return (s_acc + e_acc) / 2

    s_acc = drqa_accuracy_start(y_true, y_pred)
    e_acc = drqa_accuracy_end(y_true, y_pred)

    return _aggregate(s_acc, e_acc)
