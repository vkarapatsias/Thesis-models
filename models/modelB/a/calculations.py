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
from kernels_gen import *
from SMKL import *


class calculations():
    def __init__(self, input, output, typeOfSplitting, typeOfParams, path):

        self.input = input
        self.output = output
        self.split = typeOfSplitting
        self.param = typeOfParams
        self.path = path
        self.calc()

    def calc(self):

        source25 = self.input[:,0:39]
        source31 = self.input[:,39:78]
        source45 = self.input[:,78:117]

        target25 = self.output[:,0:4]
        target31 = self.output[:,4:8]
        target45 = self.output[:,8:12]
        target99 = self.output[:,12:self.output.shape[1]]

        ################################################################################
        ##Models
        model1 = source25
        model2 = np.concatenate((source25, source45), axis=1)
        model3 = np.concatenate((source25, source31, source45), axis=1)
        models = []
        models.append(model1)
        models.append(model2)
        models.append(model3)

        targets = [target25]
        targets.append(target45)
        targets.append(target99)
        ################################################################################
        #testing the code
        #model = source45
        #target = target45
        ##For every model
        cbks = []
        Wsub = []
        params = []
        final_estimators = []
        sizeOfTest = 0.2
        target_counter = 0
        tar = ['/25','/45','/99']
        for target in targets:
            for tarLabel in range(target.shape[1]):
                stats1 = []
                stats2 = []
                stats3 = []
                params1 = []
                params2 = []
                params3 = []
                w1 = []
                w2 = []
                w3 = []

                for www in range(10):
                    model_counter = 0
                    for model in models:
                        if (target_counter == 0) & (model_counter != 0):
                            break
                        phase = 1
                        if self.split == 0:
                            # Spliting equally per treatment
                            labels = np.asarray(range(10)) #10 samples per treatment
                            xTrainLab = []
                            xTestLab = []
                            yTrainLab = []
                            yTestLab = []
                            for ii in range(4):
                                l1,l2,l3,l4 = train_test_split(labels+ii*10, labels+ii*10, shuffle = True, test_size = sizeOfTest) #choose which 8 out of the 10 samples per treatment we will pick for the models
                                t1 = np.asarray(l1)
                                t2 = np.asarray(l2)
                                t3 = np.asarray(l3)
                                t4 = np.asarray(l4)
                                xTrainLab.append(t1)
                                xTestLab.append(t2)
                                yTrainLab.append(t3)
                                yTestLab.append(t4)
                        elif self.split == 1:
                            # Spliting equally per treatment and year
                            labels = np.asarray(range(5)) #10 samples per treatment
                            xTrainLab = []
                            xTestLab = []
                            yTrainLab = []
                            yTestLab = []
                            for ii in range(8):
                                l1,l2,l3,l4 = train_test_split(labels+ii*5, labels+ii*5, shuffle = True, test_size = sizeOfTest) #choose which 8 out of the 10 samples per treatment we will pick for the models
                                t1 = np.asarray(l1)
                                t2 = np.asarray(l2)
                                t3 = np.asarray(l3)
                                t4 = np.asarray(l4)
                                xTrainLab.append(t1)
                                xTestLab.append(t2)
                                yTrainLab.append(t3)
                                yTestLab.append(t4)

                        xTrainLab = np.asarray(xTrainLab)
                        xTestLab = np.asarray(xTestLab)
                        yTrainLab = np.asarray(yTrainLab)
                        yTestLab = np.asarray(yTestLab)

                        xTrainLab = xTrainLab.reshape(int((1-sizeOfTest)*40),)
                        yTrainLab = yTrainLab.reshape(int((1-sizeOfTest)*40),)
                        xTestLab = xTestLab.reshape(int(sizeOfTest*40),)
                        yTestLab = yTestLab.reshape(int(sizeOfTest*40),)

                        xtrain = model[xTrainLab] #(xTrainLab, numOfSources*39)
                        xtest = model[xTestLab]
                        ytrain = target[yTrainLab,tarLabel]
                        ytest = target[yTestLab,tarLabel]
                    ################################################################################
                    ## Create the multikernel for each input of the current model
                        base_estimators = []
                        for ii in range(xtrain.shape[1]):
                            ph1 = phase1(xtrain[:,ii], xtest[:,ii],ytrain,ytest,self.param)
                            temp_estim = ph1.output()
                            base_estimators.append(temp_estim)
                    # Get the combined kernels
                        current_cbks = []
                        current_Wsub = []
                        current_params = []
                        for ii in range(len(base_estimators)):
                            current_cbks.append(base_estimators[ii].smkl.get_combined_kernel())
                            current_Wsub.append(base_estimators[ii].smkl.get_kernel_weights())
                            current_params.append(base_estimators[ii].smkl.kernel_params)
                        ## KA across features
                        ph1 = phase1(xtrain, xtest, ytrain, ytest,self.param)
                        temp_estim = ph1.output()
                        base_estimators.append(temp_estim)

                        current_cbks.append(temp_estim.smkl.get_combined_kernel())
                        current_Wsub.append(temp_estim.smkl.get_kernel_weights())
                        current_params.append(temp_estim.smkl.kernel_params)

                        current_cbks = np.asarray(current_cbks)
                        current_Wsub = np.asarray(current_Wsub)
                        current_params = np.asarray(current_params)

                        cbks.append(current_cbks)
                        Wsub.append(current_Wsub)
                        params.append(current_params)
                    ################################################################################
                    ## Regression
                        ph2 = phase2(xtrain, xtest, ytrain, ytest, current_cbks, current_Wsub, current_params)
                        estimator, data = ph2.output()
                        final_estimators.append(estimator)

                        if model_counter == 0:
                            stats1.append(data[0:5])
                            params1.append(data[5:9])
                            w1.append(estimator.smkl.get_kernel_weights())

                        elif model_counter == 1:
                            stats2.append(data[0:5])
                            params2.append(data[5:9])
                            w2.append(estimator.smkl.get_kernel_weights())

                        else:
                            stats3.append(data[0:5])
                            params3.append(data[5:9])
                            w3.append(estimator.smkl.get_kernel_weights())

                        model_counter = model_counter + 1
                ################################################################################
                ## Results
                path = self.path + tar[target_counter]

                data = stats1
                with open(path + '/Stats_model1_' + str(tarLabel) + '.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerow(['Rold','Rnew','RMSE', 'CVrmse', 'MAE'])
                    writer.writerows(data)
                csvFile.close()

                data = stats2
                with open(path + '/Stats_model2_'+str(tarLabel)+'.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerow(['Rold','Rnew','RMSE', 'CVrmse', 'MAE'])
                    writer.writerows(data)
                csvFile.close()

                data = stats3
                with open(path + '/Stats_model3_'+str(tarLabel)+'.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerow(['Rold','Rnew','RMSE', 'CVrmse', 'MAE'])
                    writer.writerows(data)
                csvFile.close()
                ################################################################################
                data = params1
                with open(path + '/Parameters_model1_' + str(tarLabel) + '.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerow(['C','Epsilon','Omega', 'Kernel_alignment'])
                    writer.writerows(data)
                csvFile.close()

                data = params2
                with open(path + '/Parameters_model2_'+str(tarLabel)+'.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerow(['C','Epsilon','Omega', 'Kernel_alignment'])
                    writer.writerows(data)
                csvFile.close()

                data = params3
                with open(path + '/Parameters_model3_'+str(tarLabel)+'.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerow(['C','Epsilon','Omega', 'Kernel_alignment'])
                    writer.writerows(data)
                csvFile.close()
                ################################################################################
                data = w1
                with open(path + '/Weights_model1_' + str(tarLabel) + '.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerows(data)
                csvFile.close()

                data = w2
                with open(path + '/Weights_model2_'+str(tarLabel)+'.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerows(data)
                csvFile.close()

                data = w3
                with open(path + '/Weights_model3_'+str(tarLabel)+'.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerows(data)
                csvFile.close()
            target_counter = 1 + target_counter
