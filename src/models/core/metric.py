from tensorflow.keras.metrics import CategoricalAccuracy

###


def drqa_accuracy_start(y_true, y_pred):

    def _metric(y_true, y_pred):
        s_metric = CategoricalAccuracy()
        s_metric.update_state(y_true, y_pred)
        return s_metric.result()

    y_true_start = y_true[:, :, 0]
    y_pred_start = y_pred[:, :, 0]

    return _metric(y_true_start, y_pred_start)


def drqa_accuracy_end(y_true, y_pred):

    def _metric(y_true, y_pred):
        e_metric = CategoricalAccuracy()
        e_metric.update_state(y_true, y_pred)
        return e_metric.result()

    y_true_end = y_true[:, :, 1]
    y_pred_end = y_pred[:, :, 1]

    return _metric(y_true_end, y_pred_end)


def drqa_accuracy(y_true, y_pred):

    def _aggregate(s_acc, e_acc):
        return (s_acc + e_acc) / 2

    s_acc = drqa_accuracy_start(y_true, y_pred)
    e_acc = drqa_accuracy_end(y_true, y_pred)

    return _aggregate(s_acc, e_acc)
