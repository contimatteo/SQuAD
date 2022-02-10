from tensorflow.keras.metrics import CategoricalAccuracy  # , MeanAbsoluteError

###


def __drqa_accuracy(y_true, y_pred):
    s_metric = CategoricalAccuracy()
    s_metric.update_state(y_true, y_pred)
    return s_metric.result()


# def __drqa_mae(y_true, y_pred):
#     s_metric = MeanAbsoluteError()
#     s_metric.update_state(y_true, y_pred)
#     return s_metric.result()

###


def drqa_start_accuracy_metric(y_true, y_pred):
    return __drqa_accuracy(y_true[:, :, 0], y_pred[:, :, 0])


def drqa_end_accuracy_metric(y_true, y_pred):
    return __drqa_accuracy(y_true[:, :, 1], y_pred[:, :, 1])


# def drqa_start_mae(y_true, y_pred):
#     return __drqa_mae(y_true[:, :, 0], y_pred[:, :, 0])

# def drqa_end_mae(y_true, y_pred):
#     return __drqa_mae(y_true[:, :, 1], y_pred[:, :, 1])

###


def drqa_accuracy_metric(y_true, y_pred):

    def _aggregate(s_acc, e_acc):
        return (s_acc + e_acc) / 2

    s_acc = drqa_start_accuracy_metric(y_true, y_pred)
    e_acc = drqa_end_accuracy_metric(y_true, y_pred)

    return _aggregate(s_acc, e_acc)


# def drqa_tot_mae(y_true, y_pred):

#     def _aggregate(s_acc, e_acc):
#         return (s_acc + e_acc) / 2

#     s_mae = drqa_start_mae(y_true, y_pred)
#     e_mae = drqa_end_mae(y_true, y_pred)

#     return _aggregate(s_mae, e_mae)
