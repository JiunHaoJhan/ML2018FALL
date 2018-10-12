import numpy as np
import pandas as pd
import math
import sys
import csv 
import matplotlib
import matplotlib.pyplot as plt

PM2_5=np.load("PM2_5.npy")
normalization = np.load("normalization.npy")

def main():
    test_filename = sys.argv[1]
    output_filename = sys.argv[2]
    #print(test_filename, output_filename)
    test_x = load_test(test_filename)
    w = np.load("model.npy")
    output_answer(test_x, w, output_filename)
def load_test(test_filename):
    data = []
    for i in range(18):
        data.append([])
    pollution = 0
    test = open(test_filename ,"r", encoding = 'big5')
    rows = csv.reader(test , delimiter= ",")
    for row in rows:
        for i in range(2,11):
            if row[i] !="NR":
                data[pollution%18].append(float(row[i]))
            else:
                data[pollution%18].append(float(0))	
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
    return data

def de_normalization(predict):
    return predict*(normalization[9][1]-normalization[9][0])+normalization[9][0]

def output_answer(test_x, w, output_filename):
    test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
    #print(test_x[0])
    feature_threshold = 0.24
    #generate feature-selected train_x
    pollution = 0
    fs_test_x = test_x[:, :1]
    for term in PM2_5:
        #print(term)
        if pollution == 9:
            fs_test_x = np.concatenate((fs_test_x, test_x[:, pollution*4+1:pollution*4+10]), axis=1)
            #print(fs_test_x.shape[0], fs_test_x.shape[1])
        elif term >= feature_threshold and pollution < 9:
            fs_test_x = np.concatenate((fs_test_x, test_x[:, pollution*4+1:(pollution+1)*4+1]), axis=1)
            #print(fs_test_x.shape[0], fs_test_x.shape[1])
            #print(np.sum(test_x-fs_test_x))
        elif term >= feature_threshold and pollution > 9:
            fs_test_x = np.concatenate((fs_test_x, test_x[:, pollution*4+6:(pollution+1)*4+6]), axis=1)
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

    answer = open(output_filename, "w+")
    output = csv.writer(answer,delimiter=',',lineterminator='\n')
    output.writerow(["id","value"])
    for i in range(len(ans)):
        output.writerow(ans[i]) 
    answer.close()

main()