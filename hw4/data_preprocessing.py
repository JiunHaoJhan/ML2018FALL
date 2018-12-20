import numpy as np
import pandas as pd
import argparse, os, keras, itertools, csv, jieba, re, sys, math
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import _pickle as pk
from keras import backend as K

class DataManager:
    def __init__(self):
        self.data = {}
    def loadfile(self, train_x_path, train_y_path, test_path):
        print("Loading data...")
        train_x = []
        test_x = []
        ww = -1
        with open(train_x_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                train_x.append(row['comment'])
        train_x = self.jieba_engineering(train_x)
        # print(train_x[0])
        # exit(1)
        train_y = pd.read_csv(train_y_path)
        train_y = train_y['label'].values

        ww = -1
        with open(test_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                test_x.append(row['comment'])
        test_x = self.jieba_engineering(test_x)

        # print(len(train_x))
        # print(len(train_y))
        # print(len(test_x))

        np.save('train_x.npy', train_x)
        np.save('train_y.npy', train_y)
        np.save('test_x.npy', test_x)
        return train_x, train_y, test_x

    def word_embedding(self, train_x, test_x, embedding_size):
        print("word embedding...")
        model = Word2Vec(train_x + test_x, size=embedding_size, window=7, min_count=3, workers=4, iter=100)
        model.save("word2vec.model")

    def jieba_engineering(self, data):
        print("Jieba parsing...")
        stop_punc = ['，', ':', ';', '（', '）', '\'', '。', '-', '「', '」', ' ']
        jieba.set_dictionary('data/dict.txt.big')
        line_segs = []
        for line in data:
            line_seg = jieba.lcut(line, cut_all=False)
            line_segs.append(line_seg)
        return line_segs

    def val_set(self, all_data_x, all_data_y, percentage_val):
        print("split validation data...")
        all_data_size = all_data_x.shape[0]
        train_x_size = int(all_data_size * (1-percentage_val))

        train_x = all_data_x[:train_x_size]
        train_y = all_data_y[:train_x_size]
        val_x = all_data_x[train_x_size:]
        val_y = all_data_y[train_x_size:]

        return train_x, train_y, val_x, val_y

    def tokenize(self, vocab_size, train_x, test_x):
        print("Create new tokenizer...")
        embedding_model = Word2Vec.load('word2vec.model')
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.tokenizer.fit_on_texts(train_x+test_x)
        #save tokenizer pickle
        pk.dump(self.tokenizer, open('tokenizer', 'wb'))
        word_index = self.tokenizer.word_index
        wordvector_matrix = np.zeros((len(word_index), 300))
        for word, index in word_index.items():
            if word in embedding_model.wv.vocab:
                wordvector_matrix[index] = embedding_model.wv[word]
            #else:
            #    print("unknown word : %s" % word)
        #save word vector matrix npy
        np.save("wordvector_matrix", wordvector_matrix)

        return self.tokenizer, wordvector_matrix
    
    def padding(self, data_x, padding_size):
        print("Padding...")
        self.tokenizer = pk.load(open('tokenizer', 'rb'))
        sequences = self.tokenizer.texts_to_sequences(data_x)
        data_x_sequences = pad_sequences(sequences, maxlen=padding_size)

        return data_x_sequences
    
    def to_bow(self, data_x):
        print("To Bow...")
        self.tokenizer = pk.load(open('tokenizer', 'rb'))
        data = self.tokenizer.texts_to_matrix(data_x,mode='count')
        
        return data


    def find_padding_size(self, train_x, test_x):
        print("Find Padding size...")
        #find the mean of sequences size
        self.tokenizer = pk.load(open('tokenizer', 'rb'))
        sequences = self.tokenizer.texts_to_sequences(train_x + test_x)
        padding_size = sorted([len(i) for i in sequences])
        padding_size = np.array(padding_size)
        print(np.percentile(padding_size, [0, 10, 25, 50, 75, 90]))
        #adding_size = np.quantile(padding_size, 0.75)
        print(padding_size[900:2000])
        exit(1)

        









