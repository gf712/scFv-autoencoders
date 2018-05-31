from keras import backend as K
from keras import layers as L
import keras
import tensorflow as tf
from .defines import VH_LENGTH, VL_LENGTH
from .loss_functions import get_loss


def autoencoderV3(input_dims, latent_dim=2, cuda_device=0, RNN_cell='GRU', compile=True):

    """


    :param input_dims:
    :param latent_dim:
    :param cuda_device:
    :param RNN_cell:
    :param compile:
    :return:
    """

    # set GPU for memory allocation
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(session=sess)

    if isinstance(RNN_cell, str):
        if RNN_cell == 'GRU':
            if keras.__version__ > '2.0.9':
                # use cudnn GRU -> optimised to use GPU
                RNN = L.CuDNNGRU
            else:
                RNN = L.GRU
        elif RNN_cell == 'LSTM':
            if keras.__version__ > '2.0.9':
                RNN = L.CuDNNLSTM
            else:
                RNN = L.LSTM
        else:
            RNN = L.SimpleRNN

    elif hasattr(RNN_cell, 'mro'):
        # does this always work...?
        # this checks if RNN_cell is a child of keras.layers.RNN
        if RNN_cell.mro()[-3] == L.RNN:
            RNN = RNN_cell

    else:
        raise ValueError("Unknown RNN_cell object!")

    # define input
    input = L.Input((VL_LENGTH+VH_LENGTH, input_dims), dtype='float', name='INPUT')

    def encoder(input):

        # define first recurrent layers
        rnn_vl = RNN(32, name='VL_RNN')(input)

        # first dense layer of encoder
        dense_1 = L.Dense(32, activation='relu', name='encoder_dense_1')(rnn_vl)

        # add another layer to combine features from VL and VH
        dense_2 = L.Dense(16, activation='relu', name='encoder_dense_2')(dense_1)

        # combine dense_1 output to learn a lower dimension latent vector
        bottleneck = L.Dense(latent_dim, name='bottleneck')(dense_2)

        return bottleneck

    def decoder(encoder_layer):

        dense_r_1 = L.Dense(16, activation='relu', name='decoder_dense1')(encoder_layer)

        dense_r_2 = L.Dense(32, activation='relu', name='decoder_dense2')(dense_r_1)

        repeat_vector_r_1 = L.RepeatVector(VH_LENGTH+VL_LENGTH, name='decoder_repeatvector1')(dense_r_2 )

        rnn_r = RNN(32, return_sequences=True, name='decoder_rnn1')(repeat_vector_r_1)

        rnn_r_2 = RNN(4, return_sequences=True, name='decoder_rnn2')(rnn_r)

        output = L.TimeDistributed(L.Dense(input_dims), name='output')(rnn_r_2)

        return output

    code = encoder(input)
    reconstruction = decoder(code)

    autoencoder = keras.models.Model(inputs=input, outputs=reconstruction)
    encoder_model = keras.models.Model(inputs=input, outputs=code)

    if compile:
        masked_mse = get_loss(0)
        autoencoder.compile(optimizer=keras.optimizers.Adamax(), loss=masked_mse)

    return encoder_model, autoencoder, sess
