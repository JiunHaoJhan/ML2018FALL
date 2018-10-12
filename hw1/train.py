import numpy as np
import pandas as pd
import math
import sys
import csv 
import matplotlib
import matplotlib.pyplot as plt

PM2_5=[]
normalization=[]
for i in range(18):
    normalization.append([])

def main():
    train_x, train_y = load_train()
    print(PM2_5) 
    linear_regression(train_x, train_y)


def linear_regression(train_x, train_y):
    train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis=1)
    #cross validation
    print(train_x.shape[0], train_x.shape[1])
    batchs=36
    batch_size = int(train_x.shape[0]/batchs)
    batch_paras = [13]
    feature_paras = [0.24]
    final_RMSEloss = []
    final_batch_RMSEloss = []
    for feature_para in feature_paras:
        feature_threshold = feature_para
        #generate feature-selected train_x
        pollution = 0
        fs_train_x = train_x[:, :1]
        for term in PM2_5:
            #print(term)
            if pollution ==9:
                fs_train_x = np.concatenate((fs_train_x, train_x[:, pollution*4+1:pollution*4+10]), axis=1)
                print(fs_train_x.shape[0], fs_train_x.shape[1])
                #print(np.sum(train_x[:, :pollution*4+10]-fs_train_x))
            elif term >= feature_threshold and pollution < 9:
                fs_train_x = np.concatenate((fs_train_x, train_x[:, pollution*4+1:(pollution+1)*4+1]), axis=1)
                print(fs_train_x.shape[0], fs_train_x.shape[1])
                #print(np.sum(train_x[:, :(pollution+1)*4+1]-fs_train_x))
            elif term >= feature_threshold and pollution > 9:
                fs_train_x = np.concatenate((fs_train_x, train_x[:, pollution*4+6:(pollution+1)*4+6]), axis=1)
                print(fs_train_x.shape[0], fs_train_x.shape[1])
                #print(np.sum(train_x[:, :(pollution+1)*4+6]-fs_train_x))
            pollution+=1
        #print(np.sum(train_x-fs_train_x))
        train_x = fs_train_x
        
        batch_RMSEloss = []
        for batch in range(batchs):
            if batch == 0:
                train1_x = train_x[batch_size:]
                train1_y = train_y[batch_size:]
                val_x = train_x[:batch_size]
                val_y = train_y[:batch_size]
            elif batch == batchs-1:
                train1_x = train_x[:batch*batch_size]
                train1_y = train_y[:batch*batch_size]
                val_x = train_x[batch*batch_size:]
                val_y = train_y[batch*batch_size:]
            else:
                train1_x = train_x[:batch*batch_size]
                val_x = train_x[batch*batch_size:(batch+1)*batch_size]
                train2_x = train_x[(batch+1)*batch_size:]
                train1_x = np.concatenate((train1_x, train2_x), axis=0)
                train1_y = train_y[:batch*batch_size]
                val_y = train_y[batch*batch_size:(batch+1)*batch_size]
                train2_y = train_y[(batch+1)*batch_size:]
                train1_y = np.concatenate((train1_y, train2_y), axis=0)
                #testt = np.concatenate((val_x,train1_x), axis=0)
                #print(train1_x.shape[0])
                #print(val_x.shape[0])
                #print(np.sum(train_x-testt))
            RMSEloss = gradient_descend(train1_x, train1_y, val_x, val_y, 10000, 0.1, batch)
            batch_RMSEloss.append(RMSEloss)
        batch_RMSEloss = np.array(batch_RMSEloss)
        final_batch_RMSEloss.append(np.mean(batch_RMSEloss))
        #print(np.mean(batch_RMSEloss))

            #print(batch_RMSEloss)
        for batch_para in batch_paras:
            print("model : %f , %f " % (feature_para, batch_para))
            val_threshold = batch_para
            train_final_x=[]
            train_final_y=[]
            train_final_val_x=[]
            train_final_val_y=[]
            for term in range(len(batch_RMSEloss)):
                if batch_RMSEloss[term] <= val_threshold:
                    if train_final_x == []:
                        train_final_x = train_x[term*batch_size:(term+1)*batch_size]
                    else:
                        train_final_x = np.concatenate((train_final_x, train_x[term*batch_size:(term+1)*batch_size]), axis=0)
                    if train_final_y == []:
                        train_final_y = train_y[term*batch_size:(term+1)*batch_size]
                    else:
                        train_final_y = np.concatenate((train_final_y, train_y[term*batch_size:(term+1)*batch_size]), axis=0)
                    print(train_final_x.shape[0], train_final_x.shape[1])
                    print(train_final_y.shape[0])
            RMSEloss = gradient_descend(train_final_x, train_final_y, train_final_val_x, train_final_val_y, 10000, 0.1, -1)
            final_RMSEloss.append([RMSEloss, feature_threshold, val_threshold])
    print(final_RMSEloss)
    print(final_batch_RMSEloss)
def gradient_descend(train_x, train_y, val_x, val_y, epochs, l_rate, batch):
    w = np.zeros(len(train_x[0]))
    #print(train_x)
    train_x_t = train_x.transpose()
    prev_gra = np.zeros(len(train_x[0]))
    for i in range(epochs):
        predict = np.dot(train_x,w)
        gra = -np.dot(train_x_t, (train_y - predict))
        prev_gra += gra**2
        ada = np.sqrt(prev_gra)
        w = w - l_rate * gra/ada
    #np.save("model_%d" % (batch), w)
    RMSEloss = np.sqrt(np.mean((de_normalization(train_y) - de_normalization(predict))** 2) )
    #RMSEloss = np.sqrt(np.mean((train_y - predict)** 2) )
    # save model
    if batch == -1:
        np.save('model.npy',w)
        print("batch: %d | train error rate: %f " % (batch+1, RMSEloss))
        val_RMSEloss = RMSEloss
    else:
        val_predict = np.dot(val_x, w)
        val_RMSEloss = np.sqrt(np.mean((de_normalization(val_y) - de_normalization(val_predict))** 2) )
        #val_RMSEloss = np.sqrt(np.mean(val_y - val_predict)** 2 )
        print("batch: %d | train error rate: %f | val error rate: %f" % (batch+1, RMSEloss, val_RMSEloss))
    return val_RMSEloss
    
def load_train():
    global PM2_5
    data = []
    for i in range(18):
        data.append([])
    pollution = 0
    train = open('data/train.csv', 'r', encoding = 'big5') 
    rows = csv.reader(train, delimiter = ",")
    for row in rows:
        if pollution != 0:
            for i in range(3,27):
                if row[i] != "NR":
                    data[(pollution-1)%18].append(float(row[i]))
                else:
                    data[(pollution-1)%18].append(float(0))	
        pollution+=1
    train.close()

    df = pd.DataFrame()
    for pollution in range(18):
        df[pollution] = data[pollution]
    PM2_5 = df.corrwith(df[9])
    np.save("PM2_5.npy", PM2_5)
    '''
    plt.matshow(df.corr())
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()
    plt.show()
    '''
    #print(data[0][0])
    data = data_normalization(data, 1)
    #print(data[0][0])
    #print(data[0][0]*(normalization[0][1]-normalization[0][0])+normalization[0][0])
    train_x = []
    train_y = []
    for month in range(12):
        for n in range(471):
            train_x.append([])
            for p in range(18):
                for hr in range(9):
                    if data[p][480*month+n+hr] == 0:
                        correct_0 = 0
                        for index in range(1,4):
                            correct_hr = (hr + index) % 9
                            correct_0+=data[p][480*month+n+correct_hr]
                        data[p][480*month+n+hr] = correct_0/3
                        if data[p][480*month+n+hr] == 0:
                            data[p][480*month+n+hr] = 0.5
                    if p != 9:
                        if hr >= 5:
                            train_x[471*month+n].append(data[p][480*month+n+hr] )
                    else:
                        train_x[471*month+n].append(data[p][480*month+n+hr] )
            train_y.append(data[9][480*month+n+9])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

def load_test():
    data = []
    for i in range(18):
        data.append([])
    pollution = 0
    test = open('data/test.csv' ,"r", encoding = 'big5')
    rows = csv.reader(test , delimiter= ",")
    for row in rows:
        #if pollution %18 == 0:
            #test_x.append([])
        for i in range(2,11):
            if row[i] !="NR":
                data[pollution%18].append(float(row[i]))
                #test_x[pollution//18].append(float(row[i]))
            else:
                data[pollution%18].append(float(0))	
                #test_x[pollution//18].append(0)
        pollution+=1
    test.close()

    data = data_normalization(data, 0)
    data = np.array(data)

    test_x = []
    for n in range(data.shape[1]//9):
        test_x.append([])
        for p in range(18):
            for hr in range(9):
                if data[p][n*9+hr] == 0:
                    correct_0 = 0
                    for index in range(1,4):
                        correct_hr = (hr + index) % 9
                        correct_0+=data[p][n*9+correct_hr]
                    data[p][n*9+hr] = correct_0/3
                    if data[p][n*9+hr] == 0:
                        data[p][n*9+hr] = 0.5
                if p != 9:
                    if hr >= 5:
                        test_x[n].append(data[p][n*9+hr])
                else:
                    test_x[n].append(data[p][n*9+hr])
    test_x = np.array(test_x)
    #add square term
    #test_x = np.concatenate((test_x,test_x**2), axis=1)
    return test_x

def data_normalization(data, train):
    global normalization
    #improvable
    for pollution in range(18):
        if train:
            p_min = np.min(data[pollution], axis=0)
            p_max = np.max(data[pollution], axis=0)
            normalization[pollution].append(p_min)
            normalization[pollution].append(p_max)
        #print(normalization) 
        for i in range(len(data[pollution])):
            data[pollution][i] = (data[pollution][i] - normalization[pollution][0]) / (normalization[pollution][1] - normalization[pollution][0])
    np.save("normalization.npy",normalization)
    return data

def de_normalization(predict):
    return predict*(normalization[9][1]-normalization[9][0])+normalization[9][0]

def output_answer(test_x, w):
    test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
    #print(test_x[0])
    feature_threshold = 0.246
    #generate feature-selected train_x
    pollution = 0
    fs_test_x = test_x[:, :1]
    for term in PM2_5:
        #print(term)
        if pollution == 9:
            fs_test_x = np.concatenate((fs_test_x, test_x[:, pollution*4+1:pollution*4+10]), axis=1)
            print(fs_test_x.shape[0], fs_test_x.shape[1])
        elif term >= feature_threshold and pollution < 9:
            fs_test_x = np.concatenate((fs_test_x, test_x[:, pollution*4+1:(pollution+1)*4+1]), axis=1)
            print(fs_test_x.shape[0], fs_test_x.shape[1])
            #print(np.sum(test_x-fs_test_x))
        elif term >= feature_threshold and pollution > 9:
            fs_test_x = np.concatenate((fs_test_x, test_x[:, pollution*4+6:(pollution+1)*4+6]), axis=1)
            print(fs_test_x.shape[0], fs_test_x.shape[1])
            #print(np.sum(test_x-fs_test_x))
        pollution+=1
    #print(np.sum(test_x-fs_test_x))
    test_x = fs_test_x
    #print(test_x.shape[1])
    ans = []
    for i in range(len(test_x)):
        ans.append([])
        ans[i].append("id_" + str(i))
        predict = np.dot(w,test_x[i])
        #print(w[1], test_x[i][1], predict, de_normalization(predict))
        ans[i].append(de_normalization(predict))
        #ans[i].append(predict)

    filename = "result/predict.csv"
    answer = open(filename, "w+")
    output = csv.writer(answer,delimiter=',',lineterminator='\n')
    output.writerow(["id","value"])
    for i in range(len(ans)):
        output.writerow(ans[i]) 
    answer.close()

main()
'''
[5.847131595951411, 6.467332472049556, 6.720242637910978, 7.052816733040604, 5.693279501203789, 6.310390018176018, 6.565079655596946, 6.909165893721935, 5.676831987208386, 6.298728067285582, 6.553633047239661, 6.891906437410382, 5.5885753495456845, 6.287201409833578, 6.542705228282351, 6.879488811638106]
'''