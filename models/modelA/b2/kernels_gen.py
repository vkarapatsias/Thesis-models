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


        if len(self.xtrain.shape) == 1:
            self.xtrain = self.xtrain.reshape(-1,1) #n_samples x 1 feature
            self.xtest = self.xtest.reshape(-1,1)

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
        self.estimator.fit(self.xtrain, self.ytrain)

        sim_meas = self.estimator.smkl.KA(self.xtrain, self.ytrain)
        print('The similarity measure for KA is : ' + str(sim_meas))
        """
        combined_kernel = self.estimator.smkl.get_combined_kernel()

        r2 = self.estimator.score(self.xtest, self.ytest)
        print('R2 for current base kernel : ' + str(r2))
        self.estimator.smkl.combined_kernel = combined_kernel
        """
    def output(self):
        """
        This function will return all the variables needed.
        """
        return self.estimator

class phase2():
        """
        This class is going to generate a combined kernel for each source that will be
        used as base kernels on the next phase.

        """

        def __init__(self,xtrain, xtest, targetTrain, targetTest, kernels, weights, gammas):
            """
            Initialization of the function.

            """
            self.xtrain = xtrain
            self.ytrain = targetTrain
            self.xtest = xtest
            self.ytest = targetTest
            self.baseKernels = kernels
            self.subKernelsWeights = weights
            self.kernelParams = gammas
            self.estimator  = None
            self.calculations()

        def calculations(self):
            #matplotlib.use('Agg')
            plt.style.use('ggplot')
            rc('text', usetex=True)

            self.estimator = SMKL_SVR(Omega = 1e-4, similarity = 'KA', kernel_params = self.kernelParams, bKernels = self.baseKernels, phase = 2,bweights = self.subKernelsWeights)

            self.estimator.fit(self.xtrain, self.ytrain)

            base_align = self.estimator.smkl.KA(self.baseKernels, self.ytrain, mode = 'base', phase = 2)
            combo_align = self.estimator.smkl.score(self.baseKernels, self.ytrain, phase = 2)
            print('The combined alignment of the source (KA) is : ' + str(combo_align))
            """
            ##Plotting Results
            w = self.estimator.smkl.get_kernel_weights()
            numOfWeights = len(w)
            combined_kernel = self.estimator.smkl.get_combined_kernel()

            r2 = self.estimator.score(self.xtest, self.ytest)
            print('R2 of the combined kernel of this source  : ' + str(r2))

            self.estimator.smkl.combined_kernel = combined_kernel
            """
        def output(self):
            """
            This function will return all the variables needed.
            """
            return self.estimator

class phase3():
        """
        This class is going to generate the final combined kernel from the sources.

        """

        def __init__(self,xtrain, xtest, ytrain, ytest, source_kernels, base_kernels, source_weights, sub_weights, gammas):

            self.scbk = source_kernels
            self.Wsource = source_weights
            self.baseKernels = base_kernels
            self.Wsub = sub_weights
            self.kernelParams = gammas
            self.xtrain = xtrain    #(numOfSources,30,39)
            self.xtest = xtest      #(numOfSources,10,39)
            self.ytrain = ytrain
            self.ytest = ytest
            self.best = None
            self.SVR_obj = None
            self.old_model = None
            self.calculations()

        def calculations(self):

            plt.style.use('ggplot')
            rc('text', usetex=True)

            kernels = [self.scbk]
            kernels.append(self.baseKernels)

            W = [self.Wsource]
            W.append(self.Wsub)

            estimator = SMKL_SVR(Omega = 1e-4, similarity = 'KA', kernel_params = self.kernelParams, bKernels = kernels, bweights = W, phase = 3)

            ##Training default model for display
            estimator.fit(self.xtrain, self.ytrain)

            ##Plotting Results
            w = estimator.smkl.get_kernel_weights()
            numOfWeights = len(w)

            base_align = estimator.smkl.KA(self.scbk, self.ytrain, mode = 'base', phase = 3)
            combo_align = estimator.smkl.score(self.scbk, self.ytrain, phase = 3)
            print('The final combined alignment (KA) is : ' + str(combo_align))


            if hasattr(self.ytest, "__len__"):
            #Compare performance of combined kernel with each base kernel and test against alignment
                r2_combo = estimator.score(X = self.xtest, y = self.ytest)
                print('R2 of the combined kernel :' + str(r2_combo))

            ##Find the optimal parameters
            #Saving simple model for comparison
            self.old_model = estimator
            w_old = w
            solvers.options['show_progress'] = False

            #Grid Search for C, epsilon and Omega parameters
            C_grid = np.logspace(-2, 4, 7)
            Omega_grid = np.logspace(-4, 4, 9)
            epsilon_grid = np.logspace(-4, -1, 4)
            param_grid = {'C': C_grid, 'epsilon': epsilon_grid}

            best_score = float('inf')
            for omega in Omega_grid:
                # Create a final combined kernel for the current omega
                temp_obj = SMKL_SVR(Omega = omega, similarity = 'KA', kernel_params = self.kernelParams, bKernels = kernels, bweights = W, phase = 3)
                temp_obj.fit(self.xtrain,self.ytrain)
                kernel_train_full = temp_obj.smkl.get_combined_kernel() #30x30
                for g in ParameterGrid(param_grid):
                    current_score = 0
                    svr_obj = SVR(kernel = 'precomputed')
                    svr_obj.set_params(**g)
                    for kk in range(kernel_train_full.shape[0]):
                        # Prepare for cross validation
                        train_indexes = np.asarray(range(kernel_train_full.shape[0])) # kernel_train_full created in line  98
                        # Select the validation index
                        validation_index = kk
                        train_indexes = np.delete(train_indexes, validation_index)
                        # Training and validation
                        svr_obj.fit(kernel_train_full[train_indexes,:][:,train_indexes],self.ytrain[train_indexes])
                        y_pred = svr_obj.predict((kernel_train_full[train_indexes,:][:,validation_index].reshape(train_indexes.shape[0],1)).T)
                        #MAE
                        current_score = current_score + abs(self.ytrain[validation_index] - y_pred)
                    # save if best
                    current_score = current_score/kernel_train_full.shape[0]
                    if current_score < best_score:
                        best_score = current_score
                        best_grid = g
                        best_omega = omega

            print ("Best score: ", best_score)
            print ("Grid:", best_grid)
            self.best = SMKL_SVR(Omega = best_omega, similarity = 'KA', kernel_params = self.kernelParams, bKernels = kernels, bweights = W, phase = 3)
            self.best.set_params(**best_grid)

            ##Training best model
            self.best.fit(self.xtrain, self.ytrain)
            if hasattr(self.ytest, "__len__"):
                print('Rsquared = ' + str(self.best.score(self.xtest, self.ytest)))

            #Comparing performance with unvalidated model
            self.Rold = 0
            self.Rnew = 0
            if hasattr(self.ytest, "__len__"):
                self.Rold = self.old_model.score(self.xtest, self.ytest)
                self.Rnew = self.best.score(self.xtest, self.ytest)
                print('SMKL - SVR performance prior to parameter optimization and validation: ' + str(self.Rold))
                print('SMKL - SVR performance after parameter optimization and validation: ' + str(self.Rnew))

            a = list(best_grid.values())
            a.append(best_omega)
            a = np.asarray(a)
            if hasattr(self.ytest, "__len__"):
                rmse, CVrmse, mae, ypred = self.best.stats(self.xtest,self.ytest)
            else:
                rmse, CVrmse, mae, ypred = 0, 0, 0, self.best.predict(self.xtest)
            self.data = [self.Rold, self.Rnew, rmse, CVrmse, mae, ypred, a[0], a[1], a[2], combo_align]

        def output(self):
            """
            This function will return all the variables needed.
            """

            return self.best,self.data
