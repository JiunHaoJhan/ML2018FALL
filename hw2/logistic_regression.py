import numpy as np
import pandas as pd
import os
import sys
import argparse

def main(opt):
    train_x, train_y, test_x = loadfile(opt.train_x_path, opt.train_y_path, opt.test_x_path)
    #train_x = one_hot(train_x)
    #test_x = one_hot(test_x)
    #train_x, test_x = normalize(train_x, test_x)

    if opt.train:
        logistic_regression(train_x, train_y)
    elif opt.test:
        output_ans(test_x, opt.output_path)

def logistic_regression(train_x, train_y):
    percentage_val = 0.2
    train_x, train_y, val_x, val_y = val_set(train_x, train_y, percentage_val)

    #initail w, b, learning rate, batch size
    w = np.zeros(train_x.shape[1])
    b = np.zeros(1)
    l_rate = 0.1
    batch_size = 50
    batch_num = int(train_x.shape[0] / batch_size)
    val_period = 50 
    epochs = 1000
    loss = 0
    C0 = 0
    C1 = 0
    for i in range(len(train_y)):
        if train_y[i] == 0:
            C0+=1
        else:
            C1+=1
    y_threshhold = (float(C0)/(C0+C1))
    for epoch in range(1, epochs):
        #print(epoch)
        #savedir
        val_result = []
        if epoch % val_period == 0:
            val_x_size = val_x.shape[0]
            val_z = np.dot(val_x, w) + b
            val_predict_y = sigmoid(val_z)
            val_predict_y = np.around(sigmoid(val_z))
            for i in range(len(val_predict_y)):
                '''
                if val_predict_y[i] > y_threshhold:
                    val_predict_y[i] = 1
                else:
                    val_predict_y[i] = 0
                '''
                if(val_predict_y[i] == val_y[i]):
                    val_result.append(1)
                else:
                    val_result.append(0)
            y_threshhold = float(sum(val_result) / val_x_size) * y_threshhold + (1 - float(sum(val_result) / val_x_size)) * 0.485
            print("epoch: {} | ac rate: {}".format(epoch, float(sum(val_result) / val_x_size)), y_threshhold)

        #shuffle
        train_x, train_y = shuffle(train_x, train_y)
        loss = 0
        for batch in range(batch_num):
            train_x_batch = train_x[batch_size * batch : batch_size * (batch + 1)]
            train_y_batch = train_y[batch_size * batch : batch_size * (batch + 1)]
            #step 1.
            z = np.dot(train_x_batch, np.transpose(w)) + b
            predict_y = sigmoid(z)
            #print(train_y_batch, predict_y)
            #step 2.
            cross_entropy = - (np.dot(np.squeeze(train_y_batch), np.log(predict_y)) + np.dot((1 - np.squeeze(train_y_batch)), np.log(1-predict_y)))
            loss+=cross_entropy
            #step 3.
            #w_grad = np.sum( - (np.dot((np.squeeze(train_y_batch) - predict_y), train_x_batch)))
            w_grad = np.mean( - train_x_batch * (np.squeeze(train_y_batch) - predict_y).reshape((batch_size, 1)), axis=0)
            b_grad = np.mean( - (np.squeeze(train_y_batch) - predict_y))

            w = w - l_rate * w_grad
            b = b - l_rate * b_grad
    np.save('model_lg_w.npy', w)
    np.save('model_lg_b.npy', b)
    np.save('model_lg_y_thr.npy', y_threshhold)
    #print(w, loss)

def output_ans(test_x, output_path):
    test_x_size = test_x.shape[0]

    w = np.load('model_lg_w.npy')
    b = np.load('model_lg_b.npy')
    y_threshhold = np.load('model_lg_y_thr.npy')
    
    z = np.dot(test_x, w) + b
    predict_y = sigmoid(z)
    predict_y = np.around(sigmoid(z))
    '''
    for i in range(len(predict_y)):
        if predict_y[i] > y_threshhold:
            predict_y[i] = 1
        else:
            predict_y[i] = 0
    '''

    with open(output_path, 'w') as f:
        f.write('id,value\n')
        for i, value in enumerate(predict_y):
            f.write('id_%d,%d\n' % (i, value))
    #print(predict_y)

def one_hot(x):
    col = ['education', 'pay_0', 'pay_2', 'pay_3', 'bill1', 'bill2', 'bill3', 'pay_atm1', 'pay_atm2', 'pay_atm3']
    col1 = ['gender', 'education', 'martial_status', 'pay_0']
    df = pd.DataFrame(x)
    df.columns = col1
    #one hot pay_0
    x_onehot = pd.get_dummies(df['pay_0'], prefix = 'pay_0')
    df = df.drop('pay_0', 1)
    df = pd.concat([x_onehot, df], axis=1)
    
    x_onehot = pd.get_dummies(df['martial_status'], prefix = 'martial_status')
    df = df.drop('martial_status', 1)
    df = pd.concat([x_onehot, df], axis=1)
    
    x_onehot = pd.get_dummies(df['gender'], prefix = 'gender')
    df = df.drop('gender', 1)
    df = pd.concat([x_onehot, df], axis=1)
    
    #one hot education
    train_x_onehot = pd.get_dummies(df['education'], prefix = 'education')
    df = df.drop('education', 1)
    df = pd.concat([train_x_onehot, df], axis=1)
    print(df)
    
    df = np.array(df)
    
    #print(train_x_onehot)

    return df



def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize], Y[randomize]

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
    feature = [1, 2, 3, 5]
    feature1 = [2, 5, 6, 7, 11, 12, 13, 17, 18, 19]
    train_x = pd.read_csv(train_x_path, sep=',', header=0)
    train_x = np.array(train_x.values)
    train_x = train_x[:, feature]

    train_y = pd.read_csv(train_y_path, sep=',', header=0)
    train_y = np.array(train_y.values)

    test_x = pd.read_csv(test_x_path, sep=',', header=0)
    test_x = np.array(test_x.values)
    test_x = test_x[:, feature]
    #print(train_x[0:10])

    return train_x, train_y, test_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Logistic Regression")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', default=False, dest='train', help='input --train for training')
    group.add_argument('--test', action='store_true', default=False, dest='test', help='input --test for testing')
    parser.add_argument('--train_x_path', default='data/train_x.csv', dest='train_x_path', help='path to train_x')
    parser.add_argument('--train_y_path', default='data/train_y.csv', dest='train_y_path', help='path to train_y')
    parser.add_argument('--test_x_path', default='data/test_x.csv', dest='test_x_path', help='path to test_x')
    parser.add_argument('--output_path', default='result/predict_lg.csv', dest='output_path', help='path to output file')
    opt = parser.parse_args()
    main(opt)