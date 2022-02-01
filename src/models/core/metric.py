from tensorflow.keras.metrics import categorical_accuracy

###


def drqa_accuracy_start(y_true, y_pred):

    def _metric(y_true, y_pred):
        return categorical_accuracy(y_true, y_pred)

    y_true_start = y_true[:, :, 0]
    y_pred_start = y_pred[:, :, 0]

    return _metric(y_true_start, y_pred_start)


def drqa_accuracy_end(y_true, y_pred):

    def _metric(y_true, y_pred):
        return categorical_accuracy(y_true, y_pred)

    y_true_end = y_true[:, :, 1]
    y_pred_end = y_pred[:, :, 1]

    return _metric(y_true_end, y_pred_end)


def drqa_accuracy(y_true, y_pred):

    def _aggregate(accuracy_start, accuracy_end):
        return (accuracy_start + accuracy_end) / 2

    accuracy_start = drqa_accuracy_start(y_true, y_pred)
    accuracy_end = drqa_accuracy_end(y_true, y_pred)

    return _aggregate(accuracy_start, accuracy_end)
