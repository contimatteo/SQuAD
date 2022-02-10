import tensorflow as tf

from tensorflow.keras.losses import categorical_crossentropy  # , mae

###


def drqa_crossentropy_loss(y_true, y_pred):
    # tf.debugging.assert_all_finite(y_true, message="Loss received `nan` values.")
    # tf.debugging.assert_all_finite(y_pred, message="Loss received `nan` values.")

    def _aggregate(loss_start, loss_end):
        ### TODO: reason about the following alternatives:
        ###  - vector norm of `[loss_start, loss_end]`
        ###  - mean between `loss_start` and `loss_end`
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


def drqa_logits_loss(y_true, logits):
    # breaking the tensor into two half's to get start and end label.
    start_label = y_true[:, :, 0]
    end_label = y_true[:, :, 1]

    # braking the logits tensor into start and end part for loss calcultion.
    start_logit = logits[:, :, 0]
    end_logit = logits[:, :, 1]

    start_loss = tf.keras.backend.categorical_crossentropy(start_label, start_logit)
    end_loss = tf.keras.backend.categorical_crossentropy(end_label, end_logit)

    return start_loss + end_loss


###


def drqa_prob_sum_loss(y_true, y_pred):
    # tf.debugging.assert_all_finite(y_true, message="Loss received `nan` values.")
    # tf.debugging.assert_all_finite(y_pred, message="Loss received `nan` values.")

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
