from colorsys import yiq_to_rgb
from tensorflow.keras.metrics import CategoricalAccuracy  # , MeanAbsoluteError
from tensorflow import convert_to_tensor
import utils.configs as Configs

# np.array([a[i,:] for i in range(a.shape[0]) if a[i,-1] != 1])
###

# def __drqa_accuracy(y_true, y_pred):
#     s_metric = CategoricalAccuracy()
#     s_metric.update_state(y_true, y_pred)
#     return s_metric.result()


def __drqa_accuracy_filtered_exact_c_bit(y_true, y_pred):

    def filter_data():
        if Configs.COMPLEMENTAR_BIT == True:
            ### y_true --> (_, N_PASSAGE_TOKENS + 1)
            ### y_pred --> (_, N_PASSAGE_TOKENS + 1)

            y_true_new = convert_to_tensor(
                [y_true[i, :] for i in range(y_true.shape[0]) if y_true[i, -1] != 1]
            )
            y_pred_new = convert_to_tensor(
                [y_pred[i, :] for i in range(y_pred.shape[0]) if y_pred[i, -1] != 1]
            )

            return y_true_new, y_pred_new

        return y_true, y_pred

    y_true, y_pred = filter_data()

    if y_pred.shape[0] == 0:
        return 0

    print()
    print(y_true.shape)
    print(y_pred.shape)
    print()
    s_metric = CategoricalAccuracy()
    s_metric.update_state(y_true, y_pred)
    return s_metric.result()


def __drqa_accuracy_filtered(y_true, y_pred, accuracy_type="answer"):

    def filter_data():
        if Configs.COMPLEMENTAR_BIT == True:
            ### y_true --> (_, N_PASSAGE_TOKENS + 1)
            ### y_pred --> (_, N_PASSAGE_TOKENS + 1)

            if accuracy_type == "answer":
                y_true_index_filtered = [i for i in range(y_true.shape[0]) if y_true[i, -1] != 1]
            else:
                y_true_index_filtered = [i for i in range(y_true.shape[0]) if y_true[i, -1] == 1]

            y_true_new = convert_to_tensor([y_true[i, :] for i in y_true_index_filtered])
            y_pred_new = convert_to_tensor([y_pred[i, :] for i in y_true_index_filtered])

            return y_true_new, y_pred_new

        return y_true, y_pred

    # print("PRE FILTER#######")
    # print()
    # for i in range(y_true.shape[0]):
    #     print(y_true[i])
    # print()
    # print(y_true.shape)
    # print(y_pred.shape)
    # print()
    y_true, y_pred = filter_data()

    if y_true.shape[0] == 0:
        return 0
    # print()
    # print(y_true.shape)
    # print(y_pred.shape)
    # print("END CONTROL#######")
    s_metric = CategoricalAccuracy()
    s_metric.update_state(y_true, y_pred)
    return s_metric.result()


def drqa_start_accuracy_metric_filtered_exact_c_bit(y_true, y_pred):
    return __drqa_accuracy_filtered_exact_c_bit(y_true[:, :, 0], y_pred[:, :, 0])


def drqa_end_accuracy_metric_filtered_exact_c_bit(y_true, y_pred):
    return __drqa_accuracy_filtered_exact_c_bit(y_true[:, :, 1], y_pred[:, :, 1])


#


def drqa_start_accuracy_metric_filtered(y_true, y_pred, accuracy_type="answer"):
    return __drqa_accuracy_filtered(y_true[:, :, 0], y_pred[:, :, 0], accuracy_type)


def drqa_end_accuracy_metric_filtered(y_true, y_pred, accuracy_type="answer"):
    return __drqa_accuracy_filtered(y_true[:, :, 1], y_pred[:, :, 1], accuracy_type)


#


def drqa_acc_exact_c_bit(y_true, y_pred):

    def _aggregate(s_acc, e_acc):
        return (s_acc + e_acc) / 2

    s_acc = drqa_start_accuracy_metric_filtered_exact_c_bit(y_true, y_pred)
    e_acc = drqa_end_accuracy_metric_filtered_exact_c_bit(y_true, y_pred)

    return _aggregate(s_acc, e_acc)


def drqa_acc_answer(y_true, y_pred):

    def _aggregate(s_acc, e_acc):
        return (s_acc + e_acc) / 2

    s_acc = drqa_start_accuracy_metric_filtered(y_true, y_pred)
    e_acc = drqa_end_accuracy_metric_filtered(y_true, y_pred)

    return _aggregate(s_acc, e_acc)


# def __drqa_mae(y_true, y_pred):
#     s_metric = MeanAbsoluteError()
#     s_metric.update_state(y_true, y_pred)
#     return s_metric.result()

###

# def drqa_start_mae(y_true, y_pred):
#     return __drqa_mae(y_true[:, :, 0], y_pred[:, :, 0])

# def drqa_end_mae(y_true, y_pred):
#     return __drqa_mae(y_true[:, :, 1], y_pred[:, :, 1])

###

# def drqa_tot_mae(y_true, y_pred):

#     def _aggregate(s_acc, e_acc):
#         return (s_acc + e_acc) / 2

#     s_mae = drqa_start_mae(y_true, y_pred)
#     e_mae = drqa_end_mae(y_true, y_pred)

#     return _aggregate(s_mae, e_mae)
