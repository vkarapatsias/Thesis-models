import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import csv
import math
import os

path_list = ["results/a/ss/input/fixed", "results/a/ss/both/fixed",
             "results/a/ss/input/adaptive", "results/a/ss/both/adaptive",
             "results/b/ss/input/fixed", "results/b/ss/both/fixed",
             "results/b/ss/input/adaptive", "results/b/ss/both/adaptive",
             "results/a/rs/input/fixed", "results/a/rs/both/fixed",
             "results/a/rs/input/adaptive", "results/a/rs/both/adaptive",
             "results/b/rs/input/fixed", "results/b/rs/both/fixed",
             "results/b/rs/input/adaptive", "results/b/rs/both/adaptive"]

models = ["/25","/45","/99"]

for i in range(len(path_list)):
    for j in range(1,4):
        if j == 3:
            outputs = 10
        else:
            outputs = 4
        print("=========>",j)
        print(path_list[i]+models[j-1])
        os.mkdir(path_list[i]+models[j-1]+"/evaluation")

        if j == 1:
            upper = 2
        else:
            upper = 4

        for jj in range(1,upper):
            print("==========================\n")
            print("Current model is : "+path_list[i]+models[j-1]+"_"+ str(jj-1) +"\n")
            print("==========================\n")

            for k in range(outputs):

                sample_file = path_list[i]+models[j-1]+"/Stats_model"+str(jj)+"_"+str(k)+".csv"
                sample = pd.read_csv(sample_file, header = 0)
                sample_data = sample._get_numeric_data()
                sample_numpy_array = sample_data[0:40].as_matrix()
                sample_numpy_array = np.asarray(sample_numpy_array)

                Rold, Rnew = sample_numpy_array[:,0], sample_numpy_array[:,1]

                len = np.linspace(0,1,10)

                plt.plot(len, Rnew, label = "Rnew")
                plt.title("Optimized model's efficiency")
                legend = plt.legend(loc='upper right', fontsize='x-small')
                legend.get_frame().set_facecolor('C0')
                plt.savefig(path_list[i]+models[j-1]+"/evaluation/Rnew"+str(jj)+"_"+str(k)+"a.jpeg")
                plt.clf()

                plt.plot(len, Rnew, label = "Rnew")
                plt.plot(len,Rold, 'k--', label = "Rold")
                plt.title("Optimized vs non-optimized model's efficiency")
                legend = plt.legend(loc='upper right', fontsize='x-small')
                legend.get_frame().set_facecolor('C0')
                plt.savefig(path_list[i]+models[j-1]+"/evaluation/Comparison"+str(jj)+"_"+str(k)+"a.jpeg")
                plt.clf()
