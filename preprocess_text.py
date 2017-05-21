import nltk
import gensim
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import sys, string


if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

numerals = ['0','1','2','3','4','5','6','7','8','9']
numeral_sub_table = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
nums_zip = zip(numerals, numeral_sub_table)
#num_trans = maketrans(numerals, numeral_sub_table)


def replace_numerals(s):
    for i, num_string in nums_zip:
        s = s.replace(i, num_string)
    return s


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower:
        text = text.lower()
    text = replace_numerals(text)
    text = text.translate(maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [i for i in seq if i]

def tokenize_a_doc(s):
    try:
        tok = word_tokenize(' '.join(text_to_word_sequence(s)))
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

        self.gensimmodel = generate_embeddings(text_dataframe, columns, word_dim, w2v_workers) \
            if text_dataframe is not None else load_embeddings(word2vec_path)
        # Add a vector that encodes every word we don't find in the corpus.
        words = ["UNK"]
        words.extend(self.gensimmodel.vocab.keys())
        if save_path:
            w2v = np.load(save_path)
            vectors = w2v['vectors']
            words = w2v['words']
        else:
            vectors = get_model_vectors(word_dim, self.gensimmodel)

        self.words = words
        self.vectors = vectors
        self.word2index = words_to_indexes(words)
        self.index2word = index_to_words(words)
        self.idx2vector = index_to_vector(vectors)

    def find_word_vector(self, word_index):
        return self.idx2vector[word_index]

    def save_embeddings(self, save_name, save_path="./"):
        np.savez(save_path + "word2vec" + save_name + ".npz", vectors=self.vectors, words=self.words)

    def tokenize_data(self, sentence_array):
        return [tokenize_string(tokenize_a_doc(sent), self.word2index) for sent in sentence_array]
