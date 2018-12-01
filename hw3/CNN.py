import numpy as np
import pandas as pd
import argparse
import os
import keras
import itertools
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LeakyReLU
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

input_shapes = (0, 0, 0)
def main(opt):
    if os.path.exists('train_x.npy') and os.path.exists('train_y.npy') and os.path.exists('test_x.npy'):
        train_x = np.load('train_x.npy')
        train_y = np.load('train_y.npy')
        test_x = np.load('test_x.npy')
    else:
        train_x, train_y, test_x = loadfile(opt.train_path, opt.test_path)
    train_x, train_y, val_x, val_y = val_set(train_x, train_y, 0.05)
    #confus_mat(val_x, val_y)

    if opt.train:
        train(train_x, train_y, val_x, val_y)
    elif opt.test:
        output_ans(test_x, opt.output_path)

def confus_mat(train_x, train_y):
    #plot confusion matrix
    y=[]
    t_y=[]
    model = load_model('weights-improvement-96-0.6929.hdf5')
    predict_y = model.predict(train_x, verbose=1)
    
    for i in range(len(predict_y)):
        y.append(np.argmax(predict_y[i]))
        t_y.append(np.argmax(train_y[i]))
    conf_mat = confusion_matrix(t_y, y)
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"], normalize=True)
    plt.savefig('confusion_matrix')

def train(train_x, train_y, val_x, val_y):
    train_x_size = train_x.shape[0]
    batch_size = 32

    #model structure
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(48, 48, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(384, (3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
    
    #data augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    print(train_x.shape, train_y.shape)
    filepath="weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=120, verbose=1)
    callbacks_list = [checkpoint]


    history = model.fit_generator(
        datagen.flow(train_x, train_y, batch_size),
        steps_per_epoch=train_x_size / batch_size * 10,
        epochs=400,
        verbose=1,
        validation_data=(val_x, val_y),
        callbacks = callbacks_list
    )
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def output_ans(test_x, output_path):
    model = load_model('weights-improvement-252-0.7061.hdf5')
    predict_y_ens1 = model.predict(test_x, verbose=1)
    model = load_model('weights-improvement-206-0.7089.hdf5')
    predict_y_ens2 = model.predict(test_x, verbose=1)
    model = load_model('weights-improvement-98-0.7047.hdf5')
    predict_y_ens3 = model.predict(test_x, verbose=1)
    model = load_model('weights-improvement-207-0.7103.hdf5')
    predict_y_ens4 = model.predict(test_x, verbose=1)

    predict_y = predict_y_ens1 + predict_y_ens2 + predict_y_ens3 + predict_y_ens4
    predict_y/=4

    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, value in enumerate(predict_y):
            #print('id_%d,%d\n' % (i, np.argmax(value)))
            f.write('%d,%d\n' % (i, np.argmax(value)))

def val_set(all_data_x, all_data_y, percentage_val):
    all_data_size = all_data_x.shape[0]
    train_x_size = int(all_data_size * (1-percentage_val))

    train_x = all_data_x[:train_x_size]
    train_y = all_data_y[:train_x_size]
    val_x = all_data_x[train_x_size:]
    val_y = all_data_y[train_x_size:]

    return train_x, train_y, val_x, val_y

def loadfile(train_path, test_path):
    train_x = pd.read_csv(train_path) #, sep=',', header=0
    train_y = train_x['label']
    train_x = train_x['feature'].str.split(expand = True).astype('float32').values
    
    
    test_x = pd.read_csv(test_path, sep=',', header=0)
    test_x = test_x['feature'].str.split(expand = True).astype('float32').values

    #train_x, test_x = normalize(train_x, test_x)
    train_x /= 255
    test_x /= 255

    if K.image_data_format() == 'channels_first':
        train_x = np.reshape(train_x, (-1, 1, 48, 48))
        test_x = np.reshape(test_x, (-1, 1, 48, 48))
    else:
        train_x = np.reshape(train_x, (-1, 48, 48, 1))
        
        test_x = np.reshape(test_x, (-1, 48, 48, 1))

    train_y = keras.utils.to_categorical(np.array(train_y), 7)   

    print(train_x.shape)
    print(test_x.shape)

    np.save('train_x.npy', train_x)
    np.save('train_y.npy', train_y)
    np.save('test_x.npy', test_x)
    return train_x, train_y, test_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "CNN hw3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', default=False, dest='train', help='input --train for training')
    group.add_argument('--test', action='store_true', default=False, dest='test', help='input --test for testing')
    parser.add_argument('--train_path', default='data/train.csv', dest='train_path', help='path to train')
    parser.add_argument('--test_path', default='data/test.csv', dest='test_path', help='path to test')
    parser.add_argument('--output_path', default='predict.csv', dest='output_path', help='path to output file')
    opt = parser.parse_args()
    main(opt)