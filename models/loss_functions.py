from keras import backend as K


def get_loss(mask_value):

    """

    :param mask_value:
    :return:
    """

    mask_value = K.variable(mask_value, dtype=K.floatx())

    def masked_mse(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = K.expand_dims(1 - K.cast(mask, K.floatx()))

        loss = (y_true - y_pred) ** 2 * mask

        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)

    return masked_mse
