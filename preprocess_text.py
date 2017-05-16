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


'''
# Arguments
    x: numpy array of text data that wants to be modelled
    name: just a fancy name used to save word2vec embeddings etc
    word2vec: If None, new word2vec will be generated when calling word2vec_init.
        If a path, it will try to load existing a word2vec file (nb: custom format)
    word_dim: Dimensionality of word2vec vectors if new ones are trained.
'''


class Preprocess_text:
    def __init__(self, text_dataframe=None, columns=["text"], save_name="textdata", save_path=None, word2vec_path=None, word_dim=50, w2v_workers=3):
        # Download nltk data if it doesnt exist
        nltk.download('punkt')
        self.df = pd.DataFrame(text_dataframe, columns=columns)
        self.columns = columns

        if save_path:
            w2v = np.load(save_path)
            vectors = w2v['vectors']
            words = w2v['words']
        else:
            gensimmodel, words = self.load_gensim(text_dataframe, columns, word_dim, word2vec_path, w2v_workers)
            vectors = self.generate_new_embedding_vectors(300, gensimmodel)
            #self.idx2vector = {idx: gensimmodel.word_vec(word) for idx, word in enumerate(words)}
            #vectors = self.generate_new_embedding_vectors(word_dim, gensimmodel)
        self.words = words
        self.vectors = vectors
        self.word2index = self.words_to_indexes(words)
        self.index2word = self.index_to_words(words)
        #self.idx2vector = self.index_to_vector(vectors)
        #np.savez("word2vec" + save_name + ".npz", vectors=self.vectors, words=self.words)


        # for input column, do tokenize and keep in separate index of tokenized_row?

    @staticmethod
    def words_to_indexes(words):
        return {word: i for i, word in enumerate(words)}

    @staticmethod
    def index_to_words(words):
        return {i: word for i, word in enumerate(words)}

    @staticmethod
    def index_to_vector(vectors):
        return {i: vector for i, vector in enumerate(vectors)}

    def tokenize_data(self):
        tokenized_data = []
        for column in self.columns:
            print("Tokenizing {}".format(column))
            tokenized_data.append([tokenize_string(tokenize_a_doc(sent), self.word2index) for sent in self.df[column]])
        return tokenized_data

    @staticmethod
    def generate_new_embedding_vectors(word_dim, gensimmodel):
        unknown_vec = np.repeat(0, word_dim)
        vecs = np.vstack(map(lambda w: gensimmodel[w], gensimmodel.vocab.keys()))
        return np.vstack([unknown_vec, vecs])

    def load_gensim(self, text_dataframe, columns, word_dim, word2vec_path, w2v_workers=3):
        # Build word2vec model
        if word2vec_path:
            gensimmodel =  gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        else:
            total_sentences = [sentence for l in pd.DataFrame(text_dataframe, columns=columns).values for sentence in l]
            gensimmodel = gensim.models.word2vec.Word2Vec(total_sentences, size=word_dim, window=5, min_count=5, workers=w2v_workers)
        print('w2v model generated.')
        # Go from gensim to numpy arrays.
        # Add first index to be zero index for missing words.

        words = ["UNK"]
        words.extend(gensimmodel.vocab.keys())
        return gensimmodel, words

