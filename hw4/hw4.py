import numpy as np
import pandas as pd
import argparse, os, keras, itertools, csv, jieba, re, sys, math
from gensim.models import Word2Vec
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Input, Embedding
from keras.layers import Conv2D, MaxPooling2D, Bidirectional, LSTM, LeakyReLU, Dot
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from data_preprocessing import DataManager
import matplotlib.pyplot as plt
import _pickle as pk

def main(opt):
    dm = DataManager()
    padding_size = 100
    embedding_size = 300
    vocab_size = None

    #loading data & jieba parsing
    if os.path.exists('train_x.npy') and os.path.exists('train_y.npy') and os.path.exists('test_x.npy'):
        train_x = np.load('train_x.npy')
        train_y = np.load('train_y.npy')
        test_x = np.load('test_x.npy')
        train_x = train_x.tolist()
        test_x = test_x.tolist()
    else:
        train_x, train_y, test_x = dm.loadfile(opt.train_x_path, opt.train_y_path, opt.test_path)
    
    #word2vec
    if not os.path.exists('word2vec.model'):
        dm.word_embedding(train_x, test_x, embedding_size)
    else:
        embedding_model = Word2Vec.load("word2vec.model")

    #tokenizer
    if not os.path.exists('tokenizer') or not os.path.exists('wordvector_matrix.npy'):
        tokenizer, wordvector_matrix = dm.tokenize(vocab_size, train_x, test_x)
    else:
        tokenizer = pk.load(open('tokenizer', 'rb'))
        wordvector_matrix = np.load('wordvector_matrix.npy')

    #dm.find_padding_size(train_x, test_x)

    #training model
    if opt.train:
        #padding
        train_x = dm.padding(train_x, padding_size)

        #split validation data
        train_x, train_y, val_x, val_y = dm.val_set(train_x, train_y, 0.1)
        model = build_RNN(padding_size, tokenizer.word_index, embedding_size, wordvector_matrix)
        train(model, train_x, train_y, val_x, val_y)
    elif opt.test:
        #padding
        test_x = dm.padding(test_x, padding_size)
        output_ans(test_x, opt.output_path)

def train(model, train_x, train_y, val_x, val_y):
    batch_size = 128
    epochs = 10
    model.compile(loss='binary_crossentropy', 
                    optimizer='adam',
                    metrics=['accuracy'])
    filepath="weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = model.fit(train_x, train_y, validation_data=(val_x, val_y),
                epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.savefig('history_acc')
    # summarize history for loss
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('history_loss')
    
def output_ans(test_x, output_path):
    model = load_model('weights-improvement-002-0.7603.hdf5')
    predict_y_ens1 = model.predict(test_x, verbose=1)

    model = load_model('weights-improvement-003-0.7602.hdf5')
    predict_y_ens2 = model.predict(test_x, verbose=1)

    model = load_model('weights-improvement-003-0.7641.hdf5')
    predict_y_ens3 = model.predict(test_x, verbose=1)

    predict_y = predict_y_ens1 + predict_y_ens2 + predict_y_ens3
    predict_y/=3

    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, value in enumerate(predict_y):
            if value > 0.5:
                f.write('%d,%d\n' % (i, 1))
            else:
                f.write('%d,%d\n' % (i, 0))

def build_RNN(padding_size, word_index, wordvector_size, wordvector_matrix):
    input_first = Input(shape=(padding_size,))
    embedding = Embedding(len(word_index), output_dim=wordvector_size, weights=[wordvector_matrix], 
                            input_length=padding_size, trainable=False)(input_first)
    embedding = Bidirectional(LSTM(units=256, return_sequences=True))(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation('tanh')(embedding)

    embedding = Bidirectional(LSTM(units=512, return_sequences=True))(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation('tanh')(embedding)

    dense = Dense(64, activation='tanh')(embedding)
    dense = Dense(1, activation='relu')(dense)
    alphas = Activation('softmax')(dense)
    attention = Dot(axes=1)([alphas, embedding])
    attention = Flatten()(attention)

    dense = Dense(256)(attention)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)

    output_layer = Dense(1, activation='sigmoid')(dense)

    model = Model(input_first, output_layer)
    model.summary()

    return model
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "hw4")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', default=False, dest='train', help='input --train for training')
    group.add_argument('--test', action='store_true', default=False, dest='test', help='input --test for testing')
    parser.add_argument('--train_x_path', default='data/train_x.csv', dest='train_x_path', help='path to train x')
    parser.add_argument('--train_y_path', default='data/train_y.csv', dest='train_y_path', help='path to train y')
    parser.add_argument('--test_path', default='data/test_x.csv', dest='test_path', help='path to test')
    parser.add_argument('--dict_path', default='data/dict.txt.big', dest='dict_path', help='path to dictionary')
    parser.add_argument('--output_path', default='predict.csv', dest='output_path', help='path to output file')
    opt = parser.parse_args()
    main(opt)