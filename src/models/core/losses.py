import tensorflow as tf

from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy  # , mae

###


def __y_is_finite(y) -> bool:
    # y_shape_without_nan = (y.shape[0] * y.shape[1] * y.shape[2], )
    # return tf.boolean_mask(y, tf.math.is_finite(y)).shape == y_shape_without_nan
    try:
        tf.debugging.assert_all_finite(y, message="Loss received `nan` values.")
        return True
    except Exception:
        return False


def drqa_crossentropy_loss(y_true, y_pred):
    # tf.debugging.assert_all_finite(y_pred, message="Loss received `nan` values.")
    # if not __y_is_finite(y_pred):
    #     y_pred = tf.zeros(y_pred.shape)
    # y_pred = tf.clip_by_value(y_pred, 0, 1)
    # assert tf.reduce_min(y_pred) >= 0 and tf.reduce_max(y_pred) <= 1

    def _aggregate(loss_start, loss_end):
        return loss_start + loss_end

    def _loss(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred)
        # return tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        #  return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

    y_true_start = y_true[:, :, 0]
    y_pred_start = y_pred[:, :, 0]
    y_true_end = y_true[:, :, 1]
    y_pred_end = y_pred[:, :, 1]

    loss_start = _loss(y_true_start, y_pred_start)
    loss_end = _loss(y_true_end, y_pred_end)
    # loss_start = tf.zeros(loss_start.shape)
    # loss_end = tf.zeros(loss_end.shape)

    return _aggregate(loss_start, loss_end)


###


def drqa_prob_sum_loss(y_true, y_pred):
    #  tf.debugging.assert_all_finite(y_pred, message="Loss received `nan` values.")
    # if not __y_is_finite(y_pred):
    #     y_pred = tf.zeros(y_pred.shape)
    # y_pred = tf.clip_by_value(y_pred, .0, 1.)

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
