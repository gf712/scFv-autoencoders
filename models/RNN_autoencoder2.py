from .model_utils import check_rnn_cell
from keras import layers as L
import keras
import tensorflow as tf
from .defines import VH_LENGTH, VL_LENGTH
from .loss_functions import get_loss


def autoencoderV2(input_dims, latent_dim=2, cuda_device=0, RNN_cell='GRU', compile=True):

    """
    Two input and output autoencoders:
     - Encoder:
        - RNN(16) | RNN(16)
        - Merge(32, 4)
        - RNN(16)
        - Dense(32)
        - Dense(latent_dim)
     - Decoder:
        - Dense(32)
        - Dense(16)
        - RepeatDense(16) | RepeatDense(16)
        - RNN(16) | RNN(16)
        - TemporalDense(input_dims) | TemporalDense(input_dims)

    :param input_dims: Number of features per timestep
    :param latent_dim: Dimensions of latent vector
    :param cuda_device:
    :param RNN_cell: RNN_cell type to use. If string is provided must be 'GRU' or 'LSMT' else it is interpreted as RNN
                     Also accepts objects from classes that inherit keras.layers.RNN
    :param compile: Whether to compile the model, if yes it will use the masked MSE loss function
                    (see loss_functions.py)
    :return: autoencoder keras.models.Model object, encoder autoencoder keras.models.Model object, session tf.Session
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
        rnn_vl = RNN(16, name='VL_RNN', return_sequences=True)(inputs[0])
        rnn_vh = RNN(16, name='VH_RNN', return_sequences=True)(inputs[1])

        # merge sequences: concatenate [rnn_vl, rnn_vh]
        merge_layer = L.merge([rnn_vl, rnn_vh], concat_axis=1, mode='concat', name='merge_layer')

        rnn_1 = RNN(16, name='merged_RNN')(merge_layer)

        # add another layer to combine features from VL and VH
        dense_1 = L.Dense(32, activation='relu', name='merged_encoder_dense_1')(rnn_1)

        # combine dense_1 output to learn a lower dimension latent vector
        bottleneck = L.Dense(latent_dim, name='bottleneck')(dense_1)

        # encoder_model = keras.Model([VL_input, VH_input], bottleneck, name='encoder')

        return bottleneck

    def decoder(encoder_layer):

        dense_r_1 = L.Dense(32, activation='relu', name='merged_decoder_dense1')(encoder_layer)

        dense_r_2 = L.Dense(16, activation='relu', name='merged_decoder_dense2')(dense_r_1)

        # repeat_vector_r_1 = L.RepeatVector(VH_LENGTH + VL_LENGTH, name='decoder_repeatvector1')(dense_r_2)

        # rnn_r_1 = RNN(32, return_sequences=True, name='decoder_rnn1')(repeat_vector_r_1)

        # dense_r_3 = L.Dense(32, activation='relu', name='merged_decoder_dense2')(dense_r_2)

        outputs = []

        for name, length in zip(['VL', 'VH'], [VL_LENGTH, VH_LENGTH]):
            dense_r_3 = L.Dense(16, activation='relu', name='{}_decoder_dense1'.format(name))(dense_r_2)

            repeat_vector_r_1 = L.RepeatVector(length, name='{}_decoder_repeatvector1'.format(name))(dense_r_3)

            rnn_r = RNN(16, return_sequences=True, name='{}_decoder_rnn1'.format(name))(repeat_vector_r_1)

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
