import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from sklearn import datasets, preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.svm import SVR
import csv
from calculations import *
from kernels_gen import *
from SMKL import *

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

#Preprocessing
input_ss = preprocessing.StandardScaler().fit_transform(data_numpy_array)
input_rs = preprocessing.RobustScaler().fit_transform(data_numpy_array)

#Target
target = pd.read_csv(target_file, header = 0)
target_labels = list(target.columns.values)
target_data = target._get_numeric_data()
target_numpy_array = target_data.as_matrix()
target_numpy_array = sorted(target_numpy_array, key=lambda x:x[3]) #if we need to sort by plot
target_numpy_array = np.asarray(target_numpy_array)
target_numpy_array = target_numpy_array[:,4:target_numpy_array.shape[1]]

#Preprocessing
output = target_numpy_array
output_ss = preprocessing.StandardScaler().fit_transform(target_numpy_array)
output_rs = preprocessing.RobustScaler().fit_transform(target_numpy_array)

calculations(input_ss, output, 0, "results/ss/input/fixed")
#calculations(input_ss, output_ss, 0, "results/ss/both/fixed")
calculations(input_ss, output, 1, "results/ss/input/adaptive")
#calculations(input_ss, output_ss, 1, "results/ss/both/adaptive")

calculations(input_rs, output, 0, "results/rs/input/fixed")
#calculations(input_rs, output_rs, 0, "results/rs/both/fixed")
calculations(input_rs, output, 1, "results/rs/input/adaptive")
#calculations(input_rs, output_rs, 1, "results/rs/both/adaptive")
