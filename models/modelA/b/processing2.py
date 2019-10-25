import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import csv
import math
import os

path_list = ["results/ss/input/adaptive/", "results/ss/input/fixed/",
             "results/rs/input/adaptive/", "results/rs/input/fixed/",
             "results/none/fixed/"       , "results/none/adaptive/" ]


models = ["25","45","99"]
r2 = []
rmse = []
CVrmse = []
MAE = []

target_file = "Rice_AgronomicTraits_1617.csv"
target = pd.read_csv(target_file, header = 0)
target_labels = list(target.columns.values)
target_data = target._get_numeric_data()
target_numpy_array = target_data.as_matrix()
target_numpy_array = sorted(target_numpy_array, key=lambda x:x[3]) #if we need to sort by plot
target_numpy_array = np.asarray(target_numpy_array)
target_numpy_array = target_numpy_array[:,4:target_numpy_array.shape[1]]

target25 = target_numpy_array[:,0:4]
target45 = target_numpy_array[:,8:12]
target99 = target_numpy_array[:,12:target_numpy_array.shape[1]]

targets = [target25]
targets.append(target45)
targets.append(target99)


for i in range(len(path_list)):
    _r2 = []
    _rmse = []
    _CVrmse = []
    _mae = []

    for j in range(1,4):
        if j == 3:
            outputs = 10
        else:
            outputs = 4
        print(path_list[i]+models[j-1])
        print("model:", j)

        if j == 1:
            upper = 2
        else:
            upper = 4

        for jj in range(1,upper):
            for k in range(outputs):
                sample_file = path_list[i]+models[j-1]+"/Stats_model"+str(jj)+"_"+str(k)+".csv"
                sample = pd.read_csv(sample_file, header = 0)
                sample_data = sample.get_values()
                sample_numpy_array = sample_data[:,4]
                sample_numpy_array = np.asarray(sample_numpy_array)

                t = []
                for qq in range(len(sample_numpy_array)):
                    a = sample_numpy_array[qq][1:-1]
                    a = float(a)
                    t.append(a)

                ypred = np.asarray(t)
                y = targets[j-1][:,k]
                """
                if (i == 2) or (i == 3):
                    tar = targets[j-1][:,k]
                    mean = np.mean(tar)
                    std = np.std(tar)
                    ypred = ypred*std+mean
                elif (i == 6) or (i ==7):
                    tar = targets[j-1][:,k]
                    mean = np.mean(tar)
                    df = pd.DataFrame(tar)
                    Q1, Q3 = df.quantile(0.25,axis = 0),df.quantile(0.75,axis = 0)
                    ypred = ypred*abs(Q1[0]-Q3[0])+mean
                """
                _r2.append(round(r2_score(y, ypred),3))
                _rmse.append(round(math.sqrt(mean_squared_error(y,ypred)),3))
                temp= 100*(math.sqrt(mean_squared_error(y,ypred)))/np.mean(y)
                _CVrmse.append(round(temp,3))
                _mae.append(round(mean_absolute_error(y,ypred),3))
                print("Current target is : "+str(k+1))
    r2.append(_r2)
    rmse.append(_rmse)
    CVrmse.append(_CVrmse)
    MAE.append(_mae)

r2 = np.asarray(r2).T
rmse = np.asarray(rmse).T
CVrmse = np.asarray(CVrmse).T
MAE = np.asarray(MAE).T

data = r2
with open('evaluation/r2.csv', 'w+') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')
    writer.writerow(['scenario1','scenario2','scenario3', 'scenario4',
                     'scenario5','scenario6'])
    writer.writerows(data)
csvFile.close()

data = rmse
with open('evaluation/rmse.csv', 'w+') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')
    writer.writerow(['scenario1','scenario2','scenario3', 'scenario4',
                     'scenario5','scenario6'])
    writer.writerows(data)
csvFile.close()

data = CVrmse
with open('evaluation/CVrmse.csv', 'w+') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')
    writer.writerow(['scenario1','scenario2','scenario3', 'scenario4',
                     'scenario5','scenario6'])
    writer.writerows(data)
csvFile.close()

data = MAE
with open('evaluation/mae.csv', 'w+') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')
    writer.writerow(['scenario1','scenario2','scenario3', 'scenario4',
                     'scenario5','scenario6'])
    writer.writerows(data)
csvFile.close()
