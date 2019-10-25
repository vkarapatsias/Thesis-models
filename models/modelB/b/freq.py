import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import csv
import math
import os


path_list = ["results/rs/input/adaptive/", "results/rs/input/fixed/"]

save_to = ["rs/adaptive/", "rs/fixed/"]

models = ["25","45","99"]

for i in range(len(path_list)):
    for j in range(1,4):
        if j == 3:
            weight_files = 10
        else:
            weight_files = 4
        print("=========>",j)
        print(path_list[i]+models[j-1])

        if j == 1:
            upper = 2
        else:
            upper = 4

        for jj in range(1,upper):
            freq = []
            for k in range(weight_files):


                sample_file = path_list[i]+models[j-1]+"/Stats_model"+str(jj)+"_"+str(k)+".csv"
                sample = pd.read_csv(sample_file, header = 0)
                sample_data = sample._get_numeric_data()
                sample_numpy_array = sample_data[0:40].as_matrix()
                sample_numpy_array = np.asarray(sample_numpy_array)

                y,ypred = sample_numpy_array[:,3], sample_numpy_array[:,2]
                r2 = r2_score(y, ypred)

                if r2<0.7:
                    break

                dataset_file = "avg_indices_reflectance_1617.csv"
                dataset = pd.read_csv(dataset_file, header = 0)
                dataset_labels = list(dataset.columns.values)

                lab25 = dataset_labels[8:47]
                lab31 = dataset_labels[47:86]
                lab45 = dataset_labels[86:125]


                target_file = "Rice_AgronomicTraits_1617.csv"
                target = pd.read_csv(target_file, header = 0)
                target_labels = list(target.columns.values)
                target25 = target_labels[8:12]
                target45 = target_labels[16:20]
                target99 = target_labels[20:len(target_labels)]

                targets = [target25]
                targets.append(target45)
                targets.append(target99)

                sample_file = path_list[i]+models[j-1]+"/Weights_model"+str(jj)+"_"+str(k)+".csv"
                sample = pd.read_csv(sample_file, header = None)
                sample_data = sample._get_numeric_data()
                sample_np = sample_data[0:40].as_matrix()
                sample_np = np.asarray(sample_np)

                weights = sample_np

                indexes = np.zeros((weights.shape))
                threshold = 0.95
                for idx1 in range(weights.shape[0]):
                    sum = 0
                    temp1 = weights[idx1,:]
                    temp2 = sorted(temp1, reverse = True)
                    idx2 = 0
                    while sum < 0.95:
                        sum = sum + temp2[idx2]
                        idx2 = idx2 + 1
                    for ll in range(idx2):
                        index = np.where(temp1 == temp2[ll])
                        for lll in range(index[0].shape[0]):
                            indexes[idx1,index[lll]]  = 1

                for idx1 in range(weights.shape[0]):
                    for idx2 in range(weights.shape[1]):
                        if indexes[idx1,idx2] == 0:
                            weights[idx1,idx2] = 0

                frequency = np.zeros(weights.shape[1])
                for idx1 in range(len(frequency)):
                    frequency[idx1] = np.sum(indexes[:,idx1])
