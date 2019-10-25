import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from matplotlib import rc
from sklearn import datasets, preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.svm import SVR
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
        model1 = [source25]
        model2 = [source25]
        model2.append(source45)
        model3 = [source25]
        model3.append(source31)
        model3.append(source45)

        models = []
        models.append(model1)
        models.append(model2)
        models.append(model3)


        targets = [target25]
        targets.append(target45)
        targets.append(target99)
        ################################################################################
        #testing the code
        #model = models[1]
        #target = target45

        ##For every model
        sizeOfTest = 0.2
        target_counter = 0
        tar = ['/25','/45','/99']
        for target in targets:
            for tarLabel in range(target.shape[1]):
                final_estimators = []
                stats1 = []
                stats2 = []
                stats3 = []
                params1 = []
                params2 = []
                params3 = []
                w1 = []
                w2 = []
                w3 = []
                Wsource1 = []
                Wsource2 = []
                Wsource3 = []

                for repeats in range(10):
                    model_counter = 0
                    for model in models:
                        if (target_counter == 0) & (model_counter != 0):
                            break
                        cbks = []
                        Wsub = []
                        params = []
                        source_estimators = []
                        Xtrain = []
                        Xtest = []
                        phase = 1
                        for source in model:
                            if self.split == 0:
                                # Spliting equally per treatment
                                labels = np.asarray(range(10)) #10 samples per treatment
                                xTrainLab = []
                                xTestLab = []
                                yTrainLab = []
                                yTestLab = []
                                for ii in range(4):
                                    l1,l2,l3,l4 = train_test_split(labels+ii*10, labels+ii*10, shuffle = True, test_size = 0.2) #choose which 8 out of the 10 samples per treatment we will pick for the models
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

                            xtrain = source[xTrainLab] #(xTrainLab, numOfSources*39)
                            xtest = source[xTestLab]
                            ytrain = target[yTrainLab,tarLabel]
                            ytest = target[yTestLab,tarLabel]
                        ################################################################################
                            ## Create the multikernel for each input of the current source
                            ## KA for every feature
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

                            current_cbks = np.asarray(current_cbks) # 40x32x32
                            current_Wsub = np.asarray(current_Wsub) #40x13
                            current_params = np.asarray(current_params) #40x13

                            cbks.append(current_cbks) #(numOfSources,30,30)
                            Wsub.append(current_Wsub) #(numOfSources,39,13)
                            params.append(current_params) #(numOfSources,39,13)
                        ################################################################################
                            ## Create the multikernel of the current source
                            phase = 2
                            ph2 = phase2(xtrain,xtest,ytrain,ytest,current_cbks,current_Wsub,current_params)
                            temp = ph2.output()
                            source_estimators.append(temp)
                            print("##########################################################################")

                            Xtrain.append(xtrain)
                            Xtest.append(xtest)
                        ################################################################################
                        ## Create the multikernel
                        phase = 3
                        sources_cbk = []
                        Wsource = []
                        for i in range(len(source_estimators)):
                            sources_cbk.append(source_estimators[i].smkl.get_combined_kernel())
                            Wsource.append(source_estimators[i].smkl.get_kernel_weights())

                        ph3 = phase3(Xtrain,Xtest,ytrain,ytest,sources_cbk,cbks,Wsource,Wsub,params)
                        estimator, data = ph3.output()
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

                        for estim in source_estimators:
                            if model_counter == 0:
                                Wsource1.append(estim.smkl.get_kernel_weights())
                            elif model_counter == 1:
                                Wsource2.append(estim.smkl.get_kernel_weights())
                            else:
                                Wsource3.append(estim.smkl.get_kernel_weights())

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
                ################################################################################
                data = Wsource1
                with open(path + '/Source_Weights_model1_' + str(tarLabel) + '.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerows(data)
                csvFile.close()

                data = Wsource2
                with open(path + '/Source_Weights_model2_'+str(tarLabel)+'.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerows(data)
                csvFile.close()

                data = Wsource3
                with open(path + '/Source_Weights_model3_'+str(tarLabel)+'.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerows(data)
                csvFile.close()
            target_counter = 1 + target_counter
