from tensorflow.keras.metrics import categorical_accuracy

###


def pippo():
    pass


def pluto():
    pass


def drqa_accuracy_metric(y_true, y_pred):

    def _aggregate(accuracy_start, accuracy_end):
        return (accuracy_start + accuracy_end) / 2

    def _metric(y_true, y_pred):
        return categorical_accuracy(y_true, y_pred)

    y_true_start = y_true[:, :, 0]
    y_pred_start = y_pred[:, :, 0]

    y_true_end = y_true[:, :, 1]
    y_pred_end = y_pred[:, :, 1]

    accuracy_start = _metric(y_true_start, y_pred_start)
    accuracy_end = _metric(y_true_end, y_pred_end)

    return _aggregate(accuracy_start, accuracy_end)
