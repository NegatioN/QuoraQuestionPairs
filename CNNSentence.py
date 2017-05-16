from __future__ import print_function
import keras
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.merge import Concatenate
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

    return merge(kernels, mode='concat', concat_axis=2, name=kernel_name)

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
class CNNSentence:
    def __init__(self, max_doc_length, num_inputs, output_name_size={}, kernel_size_start=2, regularization="batch_norm",
                 embedding_vectors=[[0]]):
        self.output_name_size = output_name_size
        self.regularization = regularization
        self.dropout = 0.3
        self.output_layers = []
        self.input_layers = []
        self.model = self.create_model(num_inputs, max_doc_length, kernel_size_start, embedding_vectors)

    def create_model(self, num_inputs, max_doc_length, kernel_size_start, embedding_vectors, num_features=500):
        self.input_layers = [Input(shape=(max_doc_length,)) for i in range(num_inputs)]
        merged_inputs = Concatenate()(self.input_layers)
        x = Embedding(input_dim=len(embedding_vectors), output_dim=len(embedding_vectors[0]), weights=[embedding_vectors], name="embeddings")(merged_inputs)

        concat_layer = KernelBlock(x, max_doc_length=max_doc_length, kernel_size_start=kernel_size_start, kernel_name="flat_kernels")

        x = Flatten()(concat_layer)
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
