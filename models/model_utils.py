from keras import layers as L
import keras


def check_rnn_cell(RNN_cell):

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
        elif RNN_cell == "ResidualLSTM":
            from .residual_lstm import ResidualLSTM
            RNN = ResidualLSTM
        else:
            RNN = L.SimpleRNN

    elif hasattr(RNN_cell, 'mro'):
        # does this always work...?
        # this checks if RNN_cell is a child of keras.layers.RNN
        if RNN_cell.mro()[-3] == L.RNN:
            RNN = RNN_cell
        else:
            raise ValueError("RNN cell must be a child of keras.layers.RNN")

    else:
        raise ValueError("Unknown RNN_cell object!")

    return RNN
