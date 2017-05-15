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


def vectorize_string(wordlist, wdic):
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
    def __init__(self, text_dataframe, columns=["text"], save_name="textdata", save_path=None, word2vec_path=None, word_dim=50, w2v_workers=3):
        # Download nltk data if it doesnt exist
        nltk.download('punkt')
        self.df = pd.DataFrame(text_dataframe, columns=columns)

        if save_path:
            w2v = np.load(save_path)
            vectors = w2v['vectors']
            words = w2v['words']
        else:
            total_sentences = [sentence for l in pd.DataFrame(text_dataframe, columns=columns).values for sentence in l]
            gensimmodel, words = self.load_gensim(total_sentences, word_dim, word2vec_path, w2v_workers)
            vectors = self.generate_new_embedding_vectors(word_dim, gensimmodel)

        self.word2index = self.words_to_indexes(words)
        self.index2word = self.index_to_words(words)
        self.idx2vector = self.index_to_vector(vectors)
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

    def vectorize_data(self):
        for column in self.columns:
            print("Tokenizing %s".format(column))
            self.tokenized_data.append([tokenize_a_doc(sent) for sent in self.df[column]])
            self.df['vectorized'] = self.df.tokenized.map(lambda x: vectorize_string(x, self.word2index))

    @staticmethod
    def generate_new_embedding_vectors(word_dim, gensimmodel):
        vecs0 = np.repeat(0, word_dim)
        vecs = np.vstack(map(lambda w: gensimmodel[w], gensimmodel.wv.vocab.keys()))
        return np.vstack([vecs0, vecs])

    def load_gensim(self, sentences, word_dim, word2vec_path, w2v_workers=3):
        # Build word2vec model
        if word2vec_path:
            gensimmodel =  gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        else:
            gensimmodel = gensim.models.word2vec.Word2Vec(sentences, size=word_dim, window=5, min_count=5, workers=w2v_workers)
        print('w2v model generated.')
        # Go from gensim to numpy arrays.
        # Add first index to be zero index for missing words.
        words = ["UNK"]
        words.extend(gensimmodel.wv.vocab.keys())
        return gensimmodel, words

    def vectorize_text(self, ):
        words = self.words
        wdic = {word: i for i, word in enumerate(words)}
        self.df['vectorized'] = self.df.tokenized.map(lambda x: vectorize_string(x, wdic))

