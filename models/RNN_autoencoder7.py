from keras import layers as L
import keras
import tensorflow as tf
from .defines import VH_LENGTH, VL_LENGTH
from .loss_functions import get_loss
from .model_utils import check_rnn_cell


def autoencoderV7(input_dims, latent_dim=2, cuda_device=0, RNN_cell='GRU', compile=True):

    """


    :param input_dims:
    :param latent_dim:
    :param cuda_device:
    :param RNN_cell:
    :param compile:
    :return:
    """

    # set GPU for memory allocation
    # config = tf.ConfigProto(device_count={'GPU': cuda_device})
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    # K.set_session(sess)

    RNN = check_rnn_cell(RNN_cell)

    # define inputs
    VL_input = L.Input((VL_LENGTH, input_dims), dtype='float', name='VL_INPUT')
    VH_input = L.Input((VH_LENGTH, input_dims), dtype='float', name='VH_INPUT')

    def encoder(inputs):

        # define first recurrent layers
        rnn_vl = L.Bidirectional(RNN(16), name='VL_bidirectional_RNN', merge_mode='sum')(inputs[0])
        rnn_vh = L.Bidirectional(RNN(16), name='VH_bidirectional_RNN', merge_mode='sum')(inputs[1])

        # first dense layer of encoder
        dense_1_vl = L.Dense(32, activation='relu', name='VL_encoder_dense_1')(rnn_vl)
        dense_1_vh = L.Dense(32, activation='relu', name='VH_encoder_dense_1')(rnn_vh)

        # merge dense layers: concatenate [dense_1_vl, dense_1_vh]
        merge_layer = L.merge([dense_1_vl, dense_1_vh], mode='concat', name='merge_layer')

        # add another layer to combine features from VL and VH
        dense_1 = L.Dense(32, activation='relu', name='merged_encoder_dense_1')(merge_layer)

        # combine dense_1 output to learn a lower dimension latent vector
        bottleneck = L.Dense(latent_dim, name='bottleneck')(dense_1)

        # encoder_model = keras.Model([VL_input, VH_input], bottleneck, name='encoder')

        return bottleneck

    def decoder(encoder_layer):

        dense_r_1 = L.Dense(32, activation='relu', name='merged_decoder_dense1')(encoder_layer)

        dense_r_2 = L.Dense(64, activation='relu', name='merged_decoder_dense2')(dense_r_1)

        outputs = []

        for name, length in zip(['VL', 'VH'], [VL_LENGTH, VH_LENGTH]):
            dense_r_3 = L.Dense(16, activation='relu', name='{}_decoder_dense1'.format(name))(dense_r_2)

            repeat_vector_r_1 = L.RepeatVector(length, name='{}_decoder_repeatvector1'.format(name))(dense_r_3)

            rnn_r = L.Bidirectional(RNN(16, return_sequences=True), merge_mode='sum',
                                    name='{}_decoder_bidirectional_rnn1'.format(name))(repeat_vector_r_1)

            output_r = L.Dense(input_dims, name='{}_output'.format(name))(rnn_r)

            outputs.append(output_r)

        return outputs

    code = encoder([VL_input, VH_input])
    reconstruction = decoder(code)

    autoencoder = keras.models.Model(inputs=[VL_input, VH_input], outputs=reconstruction)
    encoder_model = keras.models.Model(inputs=[VL_input, VH_input], outputs=code)

    if compile:
        masked_mse = get_loss(0)
        autoencoder.compile(optimizer=keras.optimizers.Adamax(), loss=masked_mse)

    return encoder_model, autoencoder, sess
