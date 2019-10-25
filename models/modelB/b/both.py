import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn import datasets, preprocessing
import csv
import math
import os

path_list = [["results/rs/both/adaptive/" , "results/rs/input/adaptive/"],
             ["results/rs/both/fixed/"    , "results/rs/input/fixed/"],
             ["results/ss/both/adaptive/" , "results/ss/input/adaptive/"],
             ["results/ss/both/fixed/"    , "results/ss/input/fixed/"]]

models = ["25","45","99"]

for i in range(len(path_list)):
    for j in range(1,4):
        if j == 3:
            outputs = 9
        else:
            outputs = 4

        if j == 1:
            upper = 2
        else:
            upper = 4

        for jj in range(1,upper):
            print(("Current model is : " + str(jj-1) +"\n"))

            for k in range(outputs):

                file1 = path_list[i][0]+models[j-1]+"/Stats_model"+str(jj)+"_"+str(k)+".csv"
                file2 = path_list[i][1]+models[j-1]+"/Stats_model"+str(jj)+"_"+str(k)+".csv"

                sample = pd.read_csv(file1, header = 0)
                sample_data = sample._get_numeric_data()
                sample_numpy_array = sample_data[1:40].as_matrix()
                data1 = np.asarray(sample_numpy_array)

                sample = pd.read_csv(file2, header = 0)
                sample_data = sample._get_numeric_data()
                sample_numpy_array = sample_data[1:40].as_matrix()
                data2 = np.asarray(sample_numpy_array)

                y,ypred1,ypred2 = data2[:,3], data1[:,2], data2[:,2]

                if i<2:
                    ypred1 = preprocessing.RobustScaler().inverse_transform(ypred1)
                else:
                    ypred1 = preprocessing.StandardScaler().inverse_transform(ypred1)

                print(y)
                print(ypred1)
                print(ypred2)
                raise Exception
