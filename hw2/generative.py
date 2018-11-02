import numpy as np
import pandas as pd
import os
import sys
import argparse

def main(opt):
    train_x, train_y, test_x = loadfile(opt.train_x_path, opt.train_y_path, opt.test_x_path)
    train_x, test_x = normalize(train_x, test_x)

    if opt.train:
        generative(train_x, train_y)
    elif opt.test:
        output_ans(test_x, opt.output_path)

def generative(train_x, train_y):
    percentage_val = 0.2
    train_x, train_y, val_x, val_y = val_set(train_x, train_y, percentage_val)

    mu0=np.zeros((train_x.shape[1],))
    mu1=np.zeros((train_x.shape[1],))
    N0=0
    N1=0
    for i in range(len(train_y)):
        if train_y[i] == 0:
            mu0+=train_x[i]
            N0+=1
        elif train_y[i] == 1:
            mu1+=train_x[i]
            N1+=1
    mu0/=N0
    mu1/=N1

    sigma0 = np.zeros((train_x.shape[1], train_x.shape[1]))
    sigma1 = np.zeros((train_x.shape[1], train_x.shape[1]))
    for i in range(len(train_y)):
        if train_y[i] == 0:
            sigma0+=np.dot((np.transpose(train_x[i] - mu0)), (train_x[i] - mu0))
        elif train_y[i] == 1:
            sigma1+=np.dot((np.transpose(train_x[i] - mu1)), (train_x[i] - mu1))
    sigma0/=N0
    sigma1/=N1
    sigma = (float(N0)/(N0+N1) * sigma0 + (float(N1)/(N0+N1) * sigma1))

    para = [mu0, mu1, N0, N1]
    np.save('model_ge_sigma.npy', sigma)
    np.save('model_ge_para.npy', para)

    #val
    result_y = []
    sigma_inv = np.linalg.inv(sigma)
    w = np.dot(np.transpose(mu0 - mu1), sigma_inv)
    b = (-1/2) * np.dot(np.dot(np.transpose(mu0), sigma_inv), mu0)
    b += (1/2) * np.dot(np.dot(np.transpose(mu1), sigma_inv), mu1)
    b += np.log(float(N0)/N1)
    z = np.dot(val_x, w) + b
    predict_y = sigmoid(z)
    print(predict_y[0:10], N0/(N0+N1))
    for i in range(len(predict_y)):
        if predict_y[i] > (N0/(N0+N1)):
            predict_y[i] = 0
        else:
            predict_y[i] = 1
        if predict_y[i] == val_y[i]:
            result_y.append(1)
        else:
            result_y.append(0)
    print("ac rate: {}".format(float(sum(result_y) / len(result_y))))

def output_ans(test_x, output_path):
    sigma = np.load('model_ge_sigma.npy')
    para = np.load('model_ge_para.npy')
    mu0 = para[0]
    mu1 = para[1]
    N0 = para[2]
    N1 = para[3]

    result_y = []
    sigma_inv = np.linalg.inv(sigma)
    w = np.dot(np.transpose(mu0 - mu1), sigma_inv)
    b = (-1/2) * np.dot(np.dot(np.transpose(mu0), sigma_inv), mu0)
    b += (1/2) * np.dot(np.dot(np.transpose(mu1), sigma_inv), mu1)
    b += np.log(float(N0)/N1)
    z = np.dot(test_x, w) + b
    predict_y = sigmoid(z)
    for i in range(len(predict_y)):
        if predict_y[i] > (N0/(N0+N1)):
            predict_y[i] = 0
        else:
            predict_y[i] = 1
    with open(output_path, 'w') as f:
        f.write('id,value\n')
        for i, value in enumerate(predict_y):
            f.write('id_%d,%d\n' % (i, value))
    print(predict_y)



def normalize(train_x, test_x):
    train_test_x = np.concatenate((train_x, test_x))

    mean = sum(train_test_x) / train_test_x.shape[0]
    deviation = np.std(train_test_x, axis=0)

    mean = np.tile(mean, (train_test_x.shape[0], 1))
    deviation = np.std(train_test_x, axis=0)

    train_test_x_normed = (train_test_x - mean) / deviation

    train_x = train_test_x_normed[:train_x.shape[0]]
    test_x = train_test_x_normed[train_x.shape[0]:]

    return train_x, test_x

def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize], Y[randomize]

def sigmoid(z):
    return np.clip((1 / (1.0 + np.exp(-z))), 1e-8, 1-(1e-8))

def val_set(all_data_x, all_data_y, percentage_val):
    all_data_size = all_data_x.shape[0]
    train_x_size = int(all_data_size * (1-percentage_val))
    
    #shuffle
    all_data_x, all_data_y = shuffle(all_data_x, all_data_y)

    train_x = all_data_x[:train_x_size]
    train_y = all_data_y[:train_x_size]
    val_x = all_data_x[train_x_size:]
    val_y = all_data_y[train_x_size:]

    return train_x, train_y, val_x, val_y

def loadfile(train_x_path, train_y_path, test_x_path):

    train_x = pd.read_csv(train_x_path, sep=',', header=0)
    train_x = np.array(train_x.values)
    train_x = train_x[:, 5:6]

    train_y = pd.read_csv(train_y_path, sep=',', header=0)
    train_y = np.array(train_y.values)

    test_x = pd.read_csv(test_x_path, sep=',', header=0)
    test_x = np.array(test_x.values)
    test_x = test_x[:, 5:6]
    print(train_x)

    return train_x, train_y, test_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Logistic Regression")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', default=False, dest='train', help='input --train for training')
    group.add_argument('--test', action='store_true', default=False, dest='test', help='input --test for testing')
    parser.add_argument('--train_x_path', default='data/train_x.csv', dest='train_x_path', help='path to train_x')
    parser.add_argument('--train_y_path', default='data/train_y.csv', dest='train_y_path', help='path to train_y')
    parser.add_argument('--test_x_path', default='data/test_x.csv', dest='test_x_path', help='path to test_x')
    parser.add_argument('--output_path', default='result/predict_ge.csv', dest='output_path', help='path to output file')
    opt = parser.parse_args()
    main(opt)
