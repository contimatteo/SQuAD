from tensorflow.keras.metrics import categorical_accuracy

###


def start_accuracy(y_true, y_pred):

    def _metric(y_true, y_pred):
        return categorical_accuracy(y_true, y_pred)

    y_true_start = y_true[:, :, 0]
    y_pred_start = y_pred[:, :, 0]

    return _metric(y_true_start, y_pred_start)


def end_accuracy(y_true, y_pred):

    def _metric(y_true, y_pred):
        return categorical_accuracy(y_true, y_pred)

    y_true_end = y_true[:, :, 1]
    y_pred_end = y_pred[:, :, 1]

    return _metric(y_true_end, y_pred_end)


def tot_accuracy(y_true, y_pred):

    def _aggregate(s_acc, e_acc):
        return (s_acc + e_acc) / 2

    s_acc = start_accuracy(y_true, y_pred)
    e_acc = end_accuracy(y_true, y_pred)

    return _aggregate(s_acc, e_acc)
