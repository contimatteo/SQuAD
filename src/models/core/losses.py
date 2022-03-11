import tensorflow as tf

from tensorflow.keras.losses import categorical_crossentropy  # , mae

###


def drqa_crossentropy_loss(y_true, y_pred):

    def _aggregate(loss_start, loss_end):
        return loss_start + loss_end

    def _loss(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred)

    y_true_start = y_true[:, :, 0]
    y_pred_start = y_pred[:, :, 0]
    y_true_end = y_true[:, :, 1]
    y_pred_end = y_pred[:, :, 1]

    loss_start = _loss(y_true_start, y_pred_start)
    loss_end = _loss(y_true_end, y_pred_end)

    return _aggregate(loss_start, loss_end)


###


def drqa_prob_sum_loss(y_true, y_pred):

    def _aggregate(loss_start, loss_end):
        return loss_start + loss_end

    def _loss(y_true, y_pred):
        return tf.reduce_sum(tf.math.abs(y_true - y_pred), axis=1)

    y_true_start = y_true[:, :, 0]  ### one-hot
    y_pred_start = y_pred[:, :, 0]  ### one-hot

    y_true_end = y_true[:, :, 1]  ### one-hot
    y_pred_end = y_pred[:, :, 1]  ### one-hot

    loss_start = _loss(y_true_start, y_pred_start)
    loss_end = _loss(y_true_end, y_pred_end)

    return _aggregate(loss_start, loss_end)
