from __future__ import print_function
import keras
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Input, concatenate, LSTM, Bidirectional, RepeatVector
from keras.layers.core import *
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization


def KernelBlock(x, max_doc_length, kernel_name, kernel_size_start=3, num_kernels=3, filter_size=200):
    kernels = []
    for kernel_num in range(num_kernels):
        kernel_size = (kernel_size_start - 1) + kernel_num
        kern = Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(x)
        kernels.append(MaxPooling1D(pool_size=max_doc_length - kernel_size + 1)(kern))

    return concatenate(kernels, axis=2, name=kernel_name)


def RNNEncoder(x, dim=250, rate_drop_lstm=0.0):
    x = Bidirectional(LSTM(dim, implementation=2, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True))(x)
    return LSTM(dim, implementation=2, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=False)(x)

def RNNDecoder(x, max_doc_length, dim=250, rate_drop_lstm=0.0):
    x = RepeatVector(max_doc_length)(x)
    x = LSTM(dim, implementation=2, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)(x)
    return LSTM(dim, implementation=2, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm, return_sequences=True)(x)

class EnsambleSentence:
    def __init__(self, max_doc_length, num_inputs, output_name_size={}, final_activation="softmax", kernel_size_start=2, regularization="batch_norm",
                 embedding_length=300, num_features=500, rate_drop_lstm=0.15 + np.random.rand() * 0.25, rate_drop_dense=0.2):
        self.output_name_size = output_name_size
        self.regularization = regularization
        self.dropout = rate_drop_dense
        self.output_layers = []
        self.input_layers = []
        self.final_activation = final_activation
        self.rate_drop_lstm = rate_drop_lstm
        self.model = self.create_model(num_inputs, max_doc_length, embedding_length, kernel_size_start, num_features)

    def create_model(self, num_inputs, max_doc_length, embedding_length, kernel_size_start, num_features):
        self.input_layers = [Input(shape=(max_doc_length, embedding_length, )) for i in range(num_inputs)]
        merged_inputs = Concatenate()(self.input_layers)

        concat_layer = KernelBlock(merged_inputs, max_doc_length=max_doc_length, kernel_size_start=kernel_size_start, kernel_name="flat_kernels")

        encoder = RNNEncoder(merged_inputs, rate_drop_lstm=self.rate_drop_lstm)
        decoder = RNNDecoder(encoder, max_doc_length, rate_drop_lstm=self.rate_drop_lstm)

        cnn_flat = Flatten()(concat_layer)
        feature_layer_cnn = Dense(num_features, activation="relu", name="feature_layer_cnn")(cnn_flat)
        rnn_flat = Flatten()(decoder)
        feature_layer_rnn = Dense(num_features, activation="relu", name="feature_layer_rnn")(rnn_flat)
        cnn_dropout = Dropout(self.dropout)(feature_layer_cnn)
        rnn_dropout = Dropout(self.dropout)(feature_layer_rnn)

        merged_features = Concatenate()([cnn_dropout, rnn_dropout])
        for name, out_size in self.output_name_size.items():
            self.output_layers.append(Dense(out_size, name=name, activation=self.final_activation)(merged_features))
        return Model(self.input_layers, self.output_layers)

def get_padded_input(values, max_doc_length=15):
    return pad_sequences(values, maxlen=max_doc_length, padding='post', truncating='post')

def one_hot_y(values):
    words, unique_inverse = np.unique(values, return_inverse=True)
    return to_categorical(unique_inverse)
