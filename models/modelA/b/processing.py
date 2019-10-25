import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import csv
import math
import os

path_list = ["results/rs/both/adaptive/" , "results/rs/both/fixed/" ,
             "results/rs/input/adaptive/", "results/rs/input/fixed/",
             "results/ss/both/adaptive/" , "results/ss/both/fixed/" ,
             "results/ss/input/adaptive/", "results/ss/input/fixed/",
             "results/none/fixed/"       , "results/none/adaptive/" ]

models = ["25","45","99"]

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
    for j in range(1,4):
        if j == 3:
            outputs = 10
        else:
            outputs = 4
        print("=========>",j)
        print(path_list[i]+models[j-1])
        os.mkdir(path_list[i]+models[j-1]+"/evaluation")
        report = open(path_list[i]+models[j-1]+"/evaluation/report.txt", "a+")

        if j == 1:
            upper = 2
        else:
            upper = 4

        for jj in range(1,upper):
            report.write("==========================\n")
            report.write("Current model is : " + str(jj-1) +"\n")
            report.write("==========================\n")

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

                if (i == 0) or (i == 1):
                    tar = targets[j-1][:,k]
                    mean = np.mean(tar)
                    df = pd.DataFrame(tar)
                    Q1, Q3 = df.quantile(0.25,axis = 0),df.quantile(0.75,axis = 0)
                    ypred = ypred*abs(Q1[0]-Q3[0])+mean
                elif (i == 4) or (i == 5):
                    tar = targets[j-1][:,k]
                    mean = np.mean(tar)
                    std = np.std(tar)
                    ypred = ypred*std+mean

                r2 = r2_score(y, ypred)
                rmse = math.sqrt(mean_squared_error(y,ypred))
                CVrmse1 = 100*rmse/np.mean(ypred)
                CVrmse2 = 100*rmse/np.mean(y)
                mae = mean_absolute_error(y,ypred)
                report.write("Current target is : "+str(k+1))
                report.write("\nr2:"+str(r2)+"\nrmse:"+str(rmse)+"\nCVrmse1:"+str(CVrmse1)+"\nCVrmse2:"+str(CVrmse2)+"\nmae:"+str(mae))
                report.write("\n==========================\n")

                len1 = np.linspace(0,1,len(y))
                len2 = np.linspace(0,1,len(ypred))
                plt.plot(len1,y, label = "y")
                plt.plot(len2, ypred, 'k--', label = "ypred")
                plt.title("Predicted vs actual values")
                legend = plt.legend(loc='upper right', fontsize='x-small')
                legend.get_frame().set_facecolor('C0')
                plt.savefig(path_list[i]+models[j-1]+"/evaluation/Differences"+str(jj)+"_"+str(k)+"a.jpeg")
                plt.clf()

                plt.plot(y-ypred)
                plt.title("Residuals")
                legend.get_frame().set_facecolor('C0')
                plt.savefig(path_list[i]+models[j-1]+"/evaluation/Residuals"+str(jj)+"_"+str(k)+"b.jpeg")
                plt.clf()

        report.close()
