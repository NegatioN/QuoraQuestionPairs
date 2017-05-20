import nltk
import gensim
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd


def tokenize_a_doc(s):
    try:
        tok = word_tokenize(' '.join(s.split(" ")).lower())
    except:
        tok = ""
    return tok


def tokenize_string(wordlist, wdic):
    return [wdic.get(word) if wdic.get(word) is not None else 0 for word in wordlist]


def index_to_vector(vectors):
    return {i: vector for i, vector in enumerate(vectors)}


def words_to_indexes(words):
    return {word: i for i, word in enumerate(words)}


def index_to_words(words):
    return {i: word for i, word in enumerate(words)}


def generate_embeddings(text_dataframe, columns, word_dim, w2v_workers=3):
    total_sentences = [sentence for l in pd.DataFrame(text_dataframe, columns=columns).values for sentence in l]
    return gensim.models.word2vec.Word2Vec(total_sentences, size=word_dim, window=5, min_count=5, workers=w2v_workers)

def get_model_vectors(word_dim, gensimmodel):
    unknown_vec = np.repeat(0, word_dim)
    vecs = np.vstack(map(lambda w: gensimmodel[w], gensimmodel.vocab.keys()))
    return np.vstack([unknown_vec, vecs])

def load_embeddings(word2vec_path):
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

'''
# Arguments
    x: numpy array of text data that wants to be modelled
    name: just a fancy name used to save word2vec embeddings etc
    word2vec: If None, new word2vec will be generated when calling word2vec_init.
        If a path, it will try to load existing a word2vec file (nb: custom format)
    word_dim: Dimensionality of word2vec vectors if new ones are trained.
'''


class Preprocess_text:
    def __init__(self, text_dataframe=None, columns=["text"], save_path=None, word2vec_path=None, word_dim=300, w2v_workers=3):
        # Download nltk data if it doesnt exist
        nltk.download('punkt')
        self.df = pd.DataFrame(text_dataframe, columns=columns)

        self.gensimmodel = self.generate_embeddings(text_dataframe, columns, word_dim, w2v_workers) \
            if text_dataframe is not None else self.load_embeddings(word2vec_path)
        # Add a vector that encodes every word we don't find in the corpus.
        words = ["UNK"]
        words.extend(self.gensimmodel.vocab.keys())
        if save_path:
            w2v = np.load(save_path)
            vectors = w2v['vectors']
            words = w2v['words']
        else:
            vectors = self.get_model_vectors(word_dim, self.gensimmodel)

        self.words = words
        self.vectors = vectors
        self.word2index = words_to_indexes(words)
        self.index2word = index_to_words(words)
        self.idx2vector = index_to_vector(vectors)

    def find_word_vector(self, word):
        return self.idx2vector[self.word2index[word]]

    def save_embeddings(self, save_name, save_path="./"):
        np.savez(save_path + "word2vec" + save_name + ".npz", vectors=self.vectors, words=self.words)

    def tokenize_data(self, sentence_array):
        return [tokenize_string(tokenize_a_doc(sent), self.word2index) for sent in sentence_array]
