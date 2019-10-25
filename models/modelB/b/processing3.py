import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import csv
import math
import os

path_list = ["results/rs/input/adaptive/", "results/rs/input/fixed/"]

models = ["25","45","99"]

r2_1 = []
rmse_1 = []
CVrmse_1 = []
MAE_1 = []

r2_2 = []
rmse_2 = []
CVrmse_2 = []
MAE_2 = []

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
                sample_data = sample._get_numeric_data()
                sample_numpy_array = sample_data[0:40].as_matrix()
                sample_numpy_array = np.asarray(sample_numpy_array)

                y,ypred = sample_numpy_array[:,3], sample_numpy_array[:,2]
                """
                if (i == 2) or (i == 3):
                    tar = targets[j-1][:,k]
                    mean = np.mean(tar)
                    std = np.std(tar)
                    y,ypred = y*std+mean , ypred*std+mean
                elif (i == 6) or (i ==7):
                    tar = targets[j-1][:,k]
                    mean = np.mean(tar)
                    df = pd.DataFrame(tar)
                    Q1, Q3 = df.quantile(0.25,axis = 0),df.quantile(0.75,axis = 0)
                    y,ypred = y*abs(Q1[0]-Q3[0])+mean , ypred*abs(Q1[0]-Q3[0])+mean
                """

                _r2.append(round(r2_score(y, ypred),3))
                _rmse.append(round(math.sqrt(mean_squared_error(y,ypred)),3))
                temp= 100*(math.sqrt(mean_squared_error(y,ypred)))/np.mean(y)
                _CVrmse.append(round(temp,3))
                _mae.append(round(mean_absolute_error(y,ypred),3))
                print("Current target is : "+str(k+1))
    if i == 0:
        r2_1.append(_r2)
        rmse_1.append(_rmse)
        CVrmse_1.append(_CVrmse)
        MAE_1.append(_mae)

        r2_1 = np.asarray(r2_1).T
        rmse_1 = np.asarray(rmse_1).T
        CVrmse_1 = np.asarray(CVrmse_1).T
        MAE_1 = np.asarray(MAE_1).T

    else:
        r2_2.append(_r2)
        rmse_2.append(_rmse)
        CVrmse_2.append(_CVrmse)
        MAE_2.append(_mae)

        r2_2 = np.asarray(r2_2).T
        rmse_2 = np.asarray(rmse_2).T
        CVrmse_2 = np.asarray(CVrmse_2).T
        MAE_2 = np.asarray(MAE_2).T


data = np.concatenate((r2_1,rmse_1,CVrmse_1,MAE_1), axis = 1)
with open('evaluation/results1.csv', 'w+') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')
    writer.writerow(['R_squared','rmse','CVrmse', 'MAE'])
    writer.writerows(data)
csvFile.close()

data = np.concatenate((r2_2,rmse_2,CVrmse_2,MAE_2), axis = 1)
with open('evaluation/results2.csv', 'w+') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')
    writer.writerow(['R_squared','rmse','CVrmse', 'MAE'])
    writer.writerows(data)
csvFile.close()
