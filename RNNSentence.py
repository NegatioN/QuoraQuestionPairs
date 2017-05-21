from __future__ import print_function
import keras
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, concatenate, LSTM, Bidirectional, RepeatVector
from keras.layers.merge import Concatenate
from keras.layers.core import *
from keras.layers.normalization import BatchNormalization


def RNNEncoder(x, dim=200):
    x = Bidirectional(LSTM(dim, dropout_U=0.1, dropout_W=0.1, consume_less="gpu", return_sequences=True))(x)
    return LSTM(dim, dropout_U=0.1, dropout_W=0.1, consume_less="gpu", return_sequences=False)(x)

def RNNDecoder(x, max_doc_length, dim=200):
    x = RepeatVector(max_doc_length)(x)
    x = LSTM(dim, dropout_U=0.1, dropout_W=0.1, consume_less="gpu", return_sequences=True)(x)
    return LSTM(dim, dropout_U=0.1, dropout_W=0.1, consume_less="gpu", return_sequences=True)(x)


"""
# Arguments
    max_doc_length: Number of words to look at for model input
    output_name_size: Dictionary of {name: num_classes}-entries for your classification-layers
    kernel_size_start: Size of first kernel (number of words to look at in convolution).
        We make three kernels, where the first size is determined by this parameter,
        and the two next are sized=start+1 and size=start+2
    regularization: batchnormalisation or dropout to normalize model.
    embedding_weights: embeddings to initialize model with.
# References
    https://arxiv.org/pdf/1408.5882.pdf
"""
class RNNSentence:
    def __init__(self, max_doc_length, num_inputs, output_name_size={}, regularization="batch_norm",
                 embedding_length=300, num_features=500, dropout=0.3):
        self.output_name_size = output_name_size
        self.regularization = regularization
        self.dropout = dropout
        self.output_layers = []
        self.input_layers = []
        self.model = self.create_model(num_inputs, max_doc_length, embedding_length, num_features)

    def create_model(self, num_inputs, max_doc_length, embedding_length, num_features):
        self.input_layers = [Input(shape=(max_doc_length, embedding_length, )) for i in range(num_inputs)]
        merged_inputs = Concatenate()(self.input_layers)


        encoder = RNNEncoder(merged_inputs)
        decoder = RNNDecoder(encoder, max_doc_length)

        x = Flatten()(decoder)
        feature_layer = Dense(num_features, activation="relu", name="feature_layer")(x)
        if self.regularization == "batch_norm":
            x = BatchNormalization()(feature_layer)
        else:
            x = Dropout(self.dropout)(feature_layer)
        for name, out_size in self.output_name_size.items():
            self.output_layers.append(Dense(out_size, name=name, activation="softmax")(x))
        return Model(self.input_layers, self.output_layers)

def get_padded_input(values, max_doc_length=15):
    return pad_sequences(values, maxlen=max_doc_length, padding='post', truncating='post')

def one_hot_y(values):
    words, unique_inverse = np.unique(values, return_inverse=True)
    return to_categorical(unique_inverse)
