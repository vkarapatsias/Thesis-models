import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from sklearn import datasets, preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.svm import SVR
from SMKL import *

class phase1():
    """
    This class is going to generate a combined kernel for each feature that will be
    used as base kernels on the next phase.

    """

    def __init__(self,xtrain, xtest, targetTrain, targetTest, typeOfParams):
        """
        Initialization of the function.

        """
        self.xtrain = xtrain
        self.ytrain = targetTrain
        self.xtest = xtest
        self.ytest = targetTest
        self.estimator  = None
        self.typeOfParams = typeOfParams
        self.calculations()

    def calculations(self):

        #matplotlib.use('Agg')
        plt.style.use('ggplot')
        rc('text', usetex=True)

        if self.typeOfParams == 0:
            # Fixed range of gammas
            gammas = np.logspace(-9,3,9)
        else:
            # Gammas based on quartiles
            x1 = self.xtrain.ravel()
            x2 = x1
            diff = []
            for i in range(len(x1)):
                for j in range(len(x2)):
                    temp = x1[i]-x2[j]
                    diff.append(temp)

            df = pd.DataFrame(np.asarray(diff))
            Q1,Q2, Q3 = df.quantile(0.1,axis = 0),df.quantile(0.5,axis = 0),df.quantile(0.9,axis = 0)
            if Q1[0] < 0:
                # StandardScaler
                # Q1 is negative an Q2 is zero
                gammas = np.logspace(1/1e-2,1/Q3[0],9)
            else:
                # RobustScaler
                gammas = np.logspace(1/Q1[0],1/Q3[0],9)

        self.estimator = SMKL_SVR(Omega = 1e-4, similarity = 'KA', kernel_params = gammas)

        if len(self.xtrain.shape) == 1:
            self.xtrain = self.xtrain.reshape(-1,1) #n_samples x 1 feature

        self.estimator.fit(self.xtrain, self.ytrain)

        sim_meas = self.estimator.smkl.KA(self.xtrain, self.ytrain)
        print('The similarity measure for KA is : ' + str(sim_meas))

    def output(self):
        """
        This function will return all the variables needed.
        """
        return self.estimator

class phase2():
        """
        This class is going to generate the final combined kernel from the models.

        """

        def __init__(self,xtrain, xtest, ytrain, ytest, kernels, weights, gammas, current_sample, set, target, typeOfParams):

            self.baseKernels = kernels
            self.Wsub = weights
            self.kernelParams = gammas
            self.label = current_sample
            self.dataset = set
            self.target = target
            self.xtrain = xtrain
            self.xtest = xtest
            self.ytrain = ytrain
            self.ytest = ytest
            self.best = None
            self.SVR_obj = None
            self.old_model = None
            self.typeOfParams = typeOfParams
            self.calculations()

        def calculations(self):

            plt.style.use('ggplot')
            rc('text', usetex=True)

            estimator = SMKL_SVR(Omega = 1e-4, similarity = 'KA', kernel_params = self.kernelParams, bKernels = self.baseKernels, bweights = self.Wsub)

            ##Training default model for display
            estimator.fit(self.xtrain, self.ytrain)

            ##Plotting Results
            w = estimator.smkl.get_kernel_weights()
            numOfWeights = len(w)

            base_align = estimator.smkl.KA(self.baseKernels, self.ytrain, mode = 'base', phase = 2)
            combo_align = estimator.smkl.score(self.baseKernels, self.ytrain, phase = 2)
            # TODO: print the best base alignment of the base kernels
            print('The combined alignment (KA) of the model is : ' + str(combo_align))

            #Performance of combined kernel
            MAE_combo = estimator.score(X = self.xtest, y = self.ytest)
            print('Mean Absolute Error = ' + str(MAE_combo))

            ##Find the optimal parameters
            #Saving simple model for comparison
            self.old_model = estimator
            w_old = w
            solvers.options['show_progress'] = False

            ####################################################################
            #Grid Search for C, epsilon and Omega parameters
            C_grid = np.logspace(-2, 4, 7)
            Omega_grid = np.logspace(-4, 4, 9)
            epsilon_grid = np.logspace(-4, -1, 4)
            param_grid = {'C': C_grid,'Omega': Omega_grid, 'epsilon': epsilon_grid}

            best_score = float(0)
            for g in ParameterGrid(param_grid):
                current_score = 0
                for i in range(1):
                    #Split the dataset
                    """
                    # Only per treatment
                    C0 = np.asarray(range(10))
                    C1,C2,C3 = C0+10,C0+20,C0+30
                    if self.label < 10:
                        C0 = np.delete(C0,self.label)
                    elif self.label < 20:
                        C1 = np.delete(C1,self.label-10)
                    elif self.label < 30:
                        C2 = np.delete(C2,self.label-20)
                    else:
                        C3 = np.delete(C3,self.label-30)
                    test_treatment = np.argmin([len(C0),len(C1),len(C2),len(C3)])

                    xTrainLab = []
                    xValLab = []
                    yTrainLab = []
                    yValLab = []
                    treatment = 0
                    for labels in [C0,C1,C2,C3]:
                        if treatment == test_treatment:
                            l1,l2,l3,l4 = train_test_split(labels,labels, shuffle = True, test_size = 2/9)
                        else:
                            l1,l2,l3,l4 = train_test_split(labels,labels, shuffle = True, test_size = 3/10)

                        t1 = np.asarray(l1)
                        t2 = np.asarray(l2)
                        t3 = np.asarray(l3)
                        t4 = np.asarray(l4)
                        xTrainLab.append(t1)
                        xValLab.append(t2)
                        yTrainLab.append(t3)
                        yValLab.append(t4)
                        treatment = treatment + 1

                    xTrainLab = np.concatenate((xTrainLab[0],xTrainLab[1],xTrainLab[2],xTrainLab[3]),axis = 0)
                    xValLab = np.concatenate((xValLab[0],xValLab[1],xValLab[2],xValLab[3]),axis = 0)
                    yTrainLab = xTrainLab
                    yValLab = xValLab
                    """
                    #Both per treatment and year
                    C0_2016 = np.asarray(range(5))
                    C0_2017 ,C1_2016,C1_2017,C2_2016,C2_2017,C3_2016,C3_2017 = C0_2016+5,C0_2016+10,C0_2016+15,C0_2016+20,C0_2016+25,C0_2016+30,C0_2016+35
                    if self.label < 5:
                        C0_2016 = np.delete(C0_2016,self.label)
                    elif self.label < 10:
                        C0_2017 = np.delete(C0_2017,self.label-5)
                    elif self.label < 15:
                        C1_2016 = np.delete(C1_2016,self.label-10)
                    elif self.label < 20:
                        C1_2017 = np.delete(C1_2017,self.label-15)
                    elif self.label < 25:
                        C2_2016 = np.delete(C2_2016,self.label-20)
                    elif self.label < 30:
                        C2_2017 = np.delete(C2_2017,self.label-25)
                    elif self.label < 35:
                        C3_2016 = np.delete(C3_2016,self.label-30)
                    else:
                        C3_2017 = np.delete(C3_2017,self.label-35)
                    test_treatment = np.argmin([len(C0_2016),len(C0_2017),len(C1_2016),len(C1_2017),len(C2_2016),len(C2_2017),len(C3_2016),len(C3_2017)])

                    xTrainLab = []
                    xValLab = []
                    yTrainLab = []
                    yValLab = []
                    treatment = 0
                    for labels in [C0_2016, C0_2017,C1_2016,C1_2017,C2_2016,C2_2017,C3_2016,C3_2017]:
                        if len(labels) == 4:
                            l1,l2,l3,l4 = train_test_split(labels,labels, shuffle = True, test_size = 0.5)
                        else:
                            l1,l2,l3,l4 = train_test_split(labels,labels, shuffle = True, test_size = 0.4)

                        t1 = np.asarray(l1)
                        t2 = np.asarray(l2)
                        t3 = np.asarray(l3)
                        t4 = np.asarray(l4)
                        xTrainLab.append(t1)
                        xValLab.append(t2)
                        yTrainLab.append(t3)
                        yValLab.append(t4)
                        treatment = treatment + 1

                    xTrainLab = np.concatenate((xTrainLab[0],xTrainLab[1],xTrainLab[2],xTrainLab[3],xTrainLab[4],xTrainLab[5],xTrainLab[6],xTrainLab[7]),axis = 0)
                    xValLab = np.concatenate((xValLab[0],xValLab[1],xValLab[2],xValLab[3],xValLab[4],xValLab[5],xValLab[6],xValLab[7]),axis = 0)
                    yTrainLab = xTrainLab
                    yValLab = xValLab

                    xtrain = self.dataset[xTrainLab,:]
                    xval = self.dataset[xValLab]
                    ytrain = self.target[yTrainLab]
                    yval = self.target[yValLab]

                    #Create the combined base kernels
                    base_estimators = []
                    if self.typeOfParams == 0:
                        gammas = np.logspace(-9,3,13)
                    else:
                        # Gammas based on quartiles
                        x1 = xtrain.ravel()
                        x2 = x1
                        diff = []
                        for i in range(len(x1)):
                            for j in range(len(x2)):
                                temp = x1[i]-x2[j]
                                diff.append(temp)

                        df = pd.DataFrame(np.asarray(diff))
                        Q1,Q2, Q3 = df.quantile(0.1,axis = 0),df.quantile(0.5,axis = 0),df.quantile(0.9,axis = 0)
                        if Q1[0] < 0:
                            # StandardScaler
                            # Q1 is negative an Q2 is zero
                            gammas = np.logspace(1/1e-2,1/Q3[0],9)
                        else:
                            # RobustScaler
                            gammas = np.logspace(1/Q1[0],1/Q3[0],9)

                    for ii in range(xtrain.shape[1]):
                        temp_estimator = SMKL_SVR(Omega = 1e-4, similarity = 'KA', kernel_params = gammas)
                        if len(xtrain[:,ii].shape) == 1:
                            temp_xtrain = xtrain[:,ii].reshape(-1,1) #n_samples x 1 feature

                        temp_estimator.fit(temp_xtrain, ytrain)
                        temp_estimator.smkl.kernel_weights = self.Wsub[ii]
                        base_estimators.append(temp_estimator)

                    cbks = []
                    for ii in range(len(base_estimators)):
                        cbks.append(base_estimators[ii].smkl.get_combined_kernel())

                    temp_estimator = SMKL_SVR(Omega = 1e-4, similarity = 'KA', kernel_params = gammas)
                    temp_estimator.fit(temp_xtrain, ytrain)
                    temp_estimator.smkl.kernel_weights = self.Wsub[ii]
                    base_estimators.append(temp_estimator)

                    cbks.append(temp_estimator.smkl.get_combined_kernel())
                    cbks = np.asarray(cbks)

                    # Create the combined kernel for the current omega
                    temp_obj = SMKL_SVR(Omega = 1e-4, similarity = 'KA', kernel_params = self.kernelParams, bKernels = cbks, bweights = self.Wsub)
                    temp_obj.set_params(**g)
                    temp_obj.fit(xtrain, ytrain)
                    temp_comb_kernel = temp_obj.smkl.get_combined_kernel() #28*28
                    ypred = temp_obj.predict(xval)
                    current_score = current_score + r2_score(yval,ypred)

                #current_score = current_score/10
                if current_score > best_score:
                    best_score = current_score
                    best_grid = g

            print ("Best score: ", best_score)
            print ("Grid:", best_grid)
            self.best = SMKL_SVR(Omega = 1e-4, similarity = 'KA', kernel_params = self.kernelParams, bKernels = self.baseKernels, bweights = self.Wsub)
            self.best.set_params(**best_grid)

            ##Training best model
            self.best.fit(self.xtrain, self.ytrain)
            print('Mean Absolute Error = ' + str(self.best.score(self.xtest, self.ytest)))

            #Comparing performance with unvalidated model
            self.MAEold = self.old_model.score(self.xtest, self.ytest)
            self.MAEnew = self.best.score(self.xtest, self.ytest)

            print('SMKL - SVR performance prior to parameter optimization and validation: ' + str(self.MAEold))
            print('SMKL - SVR performance after parameter optimization and validation: ' + str(self.MAEnew))

            #rmse, CVrmse1, CVrmse2, mae = self.best.stats(self.xtest,self.ytest)
            a = list(best_grid.values())
            a = np.asarray(a)
            self.data = [self.MAEold, self.MAEnew, self.best.stats(self.xtest,self.ytest), self.ytest,a[0],a[1],a[2],combo_align]

        def output(self):
            """
            This function will return all the variables needed.
            """
            return self.best,self.data
