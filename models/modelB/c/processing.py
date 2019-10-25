import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import csv
import math
import os

path_list = ["results/ss/input/adaptive/"]

models = ["25","45","99"]

for i in range(len(path_list)):
    for j in range(1,4):
        if j == 3:
            outputs = 1
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
                sample_data = sample._get_numeric_data()
                sample_numpy_array = sample_data[0:40].as_matrix()
                sample_numpy_array = np.asarray(sample_numpy_array)

                y,ypred = sample_numpy_array[:,3], sample_numpy_array[:,2]
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
