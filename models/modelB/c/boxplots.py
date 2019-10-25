import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from sklearn import datasets, preprocessing
import csv

matplotlib.use('agg')
#rc('text', usetex=True)

## Import the dataset
dataset_file = "avg_indices_reflectance_1617.csv"
target_file = "Rice_AgronomicTraits_1617.csv"

#Features
dataset = pd.read_csv(dataset_file, header = 0)
labels = list(dataset.columns.values)
data = dataset._get_numeric_data()
data_numpy_array = data.as_matrix()
data_numpy_array = sorted(data_numpy_array, key=lambda x:x[3]) #if we need to sort by plot
data_numpy_array = np.asarray(data_numpy_array)
data_numpy_array = data_numpy_array[:,4:data_numpy_array.shape[1]]

source25 = data_numpy_array[:,0:39]
l1 = labels[8:47]
source31 = data_numpy_array[:,39:78]
l2 = labels[47:86]
source45 = data_numpy_array[:,78:117]
l3 = labels[86:125]

sources = []
sources.append(source25)
sources.append(source31)
sources.append(source45)
"""

target = pd.read_csv(target_file, header = 0)
target_labels = list(target.columns.values)
target_data = target._get_numeric_data()
target_numpy_array = target_data.as_matrix()
target_numpy_array = sorted(target_numpy_array, key=lambda x:x[3]) #if we need to sort by plot
target_numpy_array = np.asarray(target_numpy_array)
target_numpy_array = target_numpy_array[:,4:target_numpy_array.shape[1]]

target25 = target_numpy_array[:,0:4]
target31 = target_numpy_array[:,4:8]
target45 = target_numpy_array[:,8:12]
target99 = target_numpy_array[:,12:target_numpy_array.shape[1]]
target = []
target.append(target25)
target.append(target31)
target.append(target45)
target.append(target99)

l1 = target_labels[8:12]
l2 = target_labels[12:16]
l3 = target_labels[16:20]
l4 = target_labels[20:30]
"""

counter = 1
for source in sources:

    C0 = source[0:10,:]
    C1 = source[10:20,:]
    C2 = source[20:30,:]
    C3 = source[30:40,:]

    for i in range(C0.shape[1]):

        fig, ax = plt.subplots(figsize=(9, 6))
        #ax = fig.add_subplot(111)
        ax.set_xticklabels(['dfgdfg','asdasf'])
        bp = ax.boxplot([C0[0:5,i],C0[5:10,i]])

        if counter == 1:
            plt.title(l1[i])
            plt.savefig('25/C0_fig'+str(i)+'.png', bbox_inches='tight')
        elif counter == 2:
            plt.title(l2[i])
            plt.savefig('31/C0_fig'+str(i)+'.png', bbox_inches='tight')
        elif counter == 3:
            plt.title(l3[i])
            plt.savefig('45/C0_fig'+str(i)+'.png', bbox_inches='tight')
        else:
            plt.title(l4[i])
            plt.savefig('99/C0_fig'+str(i)+'.png', bbox_inches='tight')
        plt.clf()

        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot([C1[0:5,i],C1[5:10,i]])


        if counter == 1:
            plt.title(l1[i])
            plt.savefig('25/C1_fig'+str(i)+'.png', bbox_inches='tight')
        elif counter == 2:
            plt.title(l2[i])
            plt.savefig('31/C1_fig'+str(i)+'.png', bbox_inches='tight')
        elif counter == 3:
            plt.title(l3[i])
            plt.savefig('45/C1_fig'+str(i)+'.png', bbox_inches='tight')
        else:
            plt.title(l4[i])
            plt.savefig('99/C1_fig'+str(i)+'.png', bbox_inches='tight')

        plt.clf()

        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot([C2[0:5,i],C2[5:10,i]])


        if counter == 1:
            plt.title(l1[i])
            plt.savefig('25/C2_fig'+str(i)+'.png', bbox_inches='tight')
        elif counter == 2:
            plt.title(l2[i])
            plt.savefig('31/C2_fig'+str(i)+'.png', bbox_inches='tight')
        elif counter == 3:
            plt.title(l3[i])
            plt.savefig('45/C2_fig'+str(i)+'.png', bbox_inches='tight')
        else:
            plt.title(l4[i])
            plt.savefig('99/C2_fig'+str(i)+'.png', bbox_inches='tight')

        plt.clf()

        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot([C3[0:5,i],C3[5:10,i]])


        if counter == 1:
            plt.title(l1[i])
            plt.savefig('25/C3_fig'+str(i)+'.png', bbox_inches='tight')
        elif counter == 2:
            plt.title(l2[i])
            plt.savefig('31/C3_fig'+str(i)+'.png', bbox_inches='tight')
        elif counter == 3:
            plt.title(l3[i])
            plt.savefig('45/C3_fig'+str(i)+'.png', bbox_inches='tight')
        else:
            plt.title(l4[i])
            plt.savefig('99/C3_fig'+str(i)+'.png', bbox_inches='tight')

        plt.clf()

    counter= counter+1
