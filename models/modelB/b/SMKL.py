import numpy as np
import math
from sklearn.base import BaseEstimator
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn import preprocessing
from cvxopt import solvers, matrix


class SMKL(BaseEstimator):
    """
    Class implementing Similarity Based MKL


    ...

    Attributes
    ----------
    Omega : float or array-like, shape = (n_kernels, )
        Regularization parameter

    kernel : function
        Kernel Type (rbf, polynomial, etc)

    similarity : str
        Similarity measure to be optimized (KA, NKD, HSIC)

    kernel_params : array-like, shape = (n_kernels, )
        Collection of base kernel parameters

    BaseKernels : list of array-like objects, shape = (n_kernels, )
        Base kernels that form the basis of the linear combination resulting in combined kernel

    nKernels : int
        Number of base kernels

    kernel_weights : array-like, shape = (n_kernels, )
        Weights of base kernel combination

    combined_kernel : array-like, shape = (n_samples_X, n_samples_Y)
        Weighted kernel combination of base kernels


    Methods
    -------
    set_kernel_params(kernel_params)
        Set base kernel parameters.

    set_similarity_measure(similarity)
        Set appropriate similarity measure for optimization problem
        'KA'  : Kernel Alignment
        'NKD' : Norm of Kernel Difference
        'HSIC': Hilbert-Schmidt Independence Criterion

    get_kernel_weights()
        Returns weights of kernel combination.

    get_base_kernels()
        Return list of base kernels.

    make_combined_kernel(X, Z = None)
        Uses optimized kernel weights to compute kernel combination.

    get_combined_kernel()
        Returns base kernel combination.

    make_base_kernels(X)
        Create pool of base kernels.

    make_target_kernel(y)
        Create target kernel.

    KA(X = None, y = None, mode = 'combo')
        Kernel Target Alignment Calculation.

    NKD(X = None, y = None, mode = 'combo')
        Norm of Kernel Difference Calculation.

    HSIC(X = None, y = None, mode = 'combo')
        Hilbert-Schmidt Independence Criterion Calculation.

    fit(X, y)
        Function that solves quadratic optimization problem based on similarity measure.

    score(X, y)
        Calculate similarity measure between combined kernel and target kernel.
    """


    def __init__(self, Omega = 1, kernel = rbf_kernel, similarity = 'NKD', kernel_params = None, baseKernels = None):
        """
        Parameters
        ----------
        Omega : float or array-like, shape = (n_kernels, )
            Regularization parameter

        kernel : function
            Kernel Type (rbf, polynomial, etc)

        similarity : str
            Similarity measure to be optimized (KA, NKD, HSIC)

        kernel_params : array-like, shape = (n_kernels, )
            Collection of base kernel parameters
        """

        if(similarity is not 'KA' and similarity is not 'NKD' and similarity is not 'HSIC'):
            raise Exception('Not a valid similarity measure.')
        self.Omega = Omega
        self.kernel = kernel
        self.similarity = similarity
        self.kernel_params = kernel_params
        if(self.kernel_params is not None):
            self.nKernels = len(self.kernel_params)
        self.BaseKernels = baseKernels
        if self.BaseKernels is not None:
            self.nKernels = self.BaseKernels.shape[0]
        self.kernel_weights = None
        self.combined_kernel = None
        self.Kyy = None



    def set_kernel_params(self, kernel_params):
        """"
        Set base kernel parameters.

        Parameters
        -----------
        kernel_params : array-like, shape = (n_kernels, )
            Collection of base kernel parameters (degrees for polynomial - inverse widths for rbf)

        Returns
        -------
        self : object
        """

        self.kernel_params = kernel_params
        self.nKernels = len(self.kernel_params)
        return self



    def set_similarity_measure(self, similarity):
        """"
        Set appropriate similarity measure for optimization problem
        'KA' : Kernel Alignment
        'NKD': Norm of Kernel Difference
        'HSIC: Hilbert-Schmidt Independence Criterion

        Parameters
        ----------
        similarity : str
            Similarity measure to be optimized (KA, NKD, HSIC)

        Returns
        -------
        self : object

        Raises
        ------
        Exception
            If argument passed is not one of the three above, an exception is raised.
        """

        if(similarity is not 'KA' or similarity is not 'NKD' or similarity is not 'HSIC'):
            raise Exception('Invalid similarity measure')
        self.similarity = similarity
        return self



    def get_kernel_weights(self):
        """
        Returns weights of kernel combination.

        Returns
        -------
        w : array-like, shape = (n_kernels, )
            Collection of base kernel parameters

        Raises
        ------
        Exception
            An exception is raised if the function is called before the optimization process yields the kernel weights.
        """

        if self.kernel_weights is not None:
            return self.kernel_weights
        else:
            raise Exception('Kernel weights have not been computed yet.')



    def get_base_kernels(self):
        """
        Return list of base kernels.

        Returns
        -------
        kernels : list of array-like objects, shape = (n_kernels, )
            Base kernels that form the basis of the linear combination resulting in combined kernel

        Raises
        ------
        Exception
            An exception is raised if the function is called before the function self.make_base_kernels()
        """

        if self.BaseKernels is not None:
            return self.BaseKernels
        else:
            raise Exception('Base Kernels have not been computed yet.')



    def make_combined_kernel(self, X, Z = None, phase = 1, Wsub = None):
        """
        Uses optimized kernel weights to compute kernel combination.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features_X)
            Features on which kernel will be computed

        Z : array, shape = (n_samples, n_features_Z)
            Test features - supplied when combined kernel is to be used for prediction

        Returns
        -------
        self : object
        """

        if self.kernel_weights is not None:
            if Z is None:
                if phase == 1:
                    self.make_base_kernels(X)
                    """
                    else is not implemented since the Base Kernels are passed in SMKL_SVR object
                    with initialization.
                    """
            else:
                if phase == 1:
                    self.make_base_kernels(X, Z)
                else:
                    self.BaseKernels = []
                    for i in range(self.kernel_params.shape[0]): #for each feature
                        if i == self.kernel_params.shape[0]-1: #for the multikernel across features
                            temp = [self.kernel(X, Z, param) for param in self.kernel_params[i]] #for each gamma parameter
                        else:
                            temp = [self.kernel(X[:,i].reshape(-1,1), Z[:,i].reshape(-1,1), param) for param in self.kernel_params[i]] #for each gamma parameter

                        temp = np.dstack(temp).dot(Wsub[i]) #multiply with the weights in each feature
                        self.BaseKernels.append(temp)
            self.combined_kernel = np.dstack(self.BaseKernels).dot(self.kernel_weights)
        else:
            print('Kernel weights have not been computed yet\n Cannot produce kernel combination.')
        return self



    def get_combined_kernel(self):
        """
        Returns base kernel combination.

        Returns
        -------
        ckernel : array, shape = (n_samples_X, n_samples_Y)
            Weighted kernel combination of base kernels
        """

        if self.combined_kernel is not None:
            return self.combined_kernel
        else:
            print('Kernel weights have not been computed yet\n Cannot produce kernel combination.')



    def make_base_kernels(self, X, Z = None):
        """
        Create pool of base kernels.

        Parameters
        ----------
        X : array, shape = (n_samples_X, n_features)
            features on which kernel will be computed

        Z : array, shape = (n_samples_Z, n_features)
            test features - supplied when combined kernel is to be used for prediction

        Return
        ------
        self : object
        """

        if self.kernel_params is None:
            raise Exception('Kernel parameters have not been set.')
        if(Z is None):
            self.BaseKernels = [self.kernel(X, X, param) for param in self.kernel_params]
        else:
            self.BaseKernels = [self.kernel(X, Z, param) for param in self.kernel_params]
        return self



    def make_target_kernel(self, y):
        """
        Create target kernel.

        Parameters
        ----------
        y : array-like, shape = (n_samples, )
            target variables

        Returns
        -------
        self : object
        """

        nSamples = y.size
        self.Kyy = linear_kernel(np.expand_dims(y, axis = 1)) + (1e-9) * np.eye(nSamples)
        return self



    def _get_centered_kernel(self, kernel):
        """
        Preprocessing function that centers a kernel matrix.
        """

        nSamples = kernel.shape[0]
        H = np.eye(nSamples) - np.ones(shape = (nSamples, 1)).dot(np.ones(shape = (nSamples, 1)).T) / nSamples
        Kc = H.dot(kernel.dot(H))
        D = np.expand_dims(np.diag(Kc), axis = 1)
        return Kc / np.sqrt(D.dot(D.T))



    def _get_optim_params(self):
        """
        Helper function that computes the corresponding matrices and vectors to be plugged to the optimization problem.
        """

        if self.similarity is 'KA' or self.similarity is 'NKD':
            self.__A = np.asarray([np.sum(self.BaseKernels[i] * self.Kyy) for i in range(self.nKernels)])
            B = np.zeros(shape = (self.nKernels, self.nKernels))
            for i in range(self.nKernels):
                for j in range(i, self.nKernels):
                    B[i,j] = np.sum(self.BaseKernels[i] * self.BaseKernels[j])
            self.__B = B + B.T - np.diag(np.diag(B))
        else:
            nSamples = self.Kyy.shape[0]
            self.__C =  np.asarray([np.sum(self._get_centered_kernel(self.BaseKernels[i]) * self.Kyy) for i in range(self.nKernels)]) / (nSamples)**2



    def _KA(self):
        """
        Calculate Kernel Alginment between combined kernel and target kernel.
        """

        num = np.sum(self.combined_kernel * self.Kyy)
        den = np.linalg.norm(self.combined_kernel, 'fro') * np.linalg.norm(self.Kyy, 'fro')
        return num / den



    def KA(self, X = None, y = None, mode = 'combo', phase = 1):
        """
        Kernel Target Alignment Calculation.

        Parameters
        ----------
        mode : str
            'combo' for computing KTA of combined kernel with target kernel
            'base' for computing KTA of each base kernel with target kernel

        X : array, shape = (n_samples_X, n_features)
            Feature vector for computing base kernels' alignment

        y : array, shape = (n_samples_X)
            Target variable for computing base kernels' alignment

        Returns
        -------
        align : array-like, shape = (n_kernels, )

        Raises
        ------
        Exception
            An exception is raised if an invalid mode is passed as argument
        """

        if mode is 'combo':
            return self._KA()
        elif mode is 'base':
            if X is None or y is None:
                raise Exception('Error: Required data not supplied')
            if phase == 1:
                self.make_base_kernels(X)
            self.make_target_kernel(y)
            alignment = []
            for kernel in self.BaseKernels:
                num = np.sum(kernel * self.Kyy)
                den = np.linalg.norm(kernel, 'fro') * np.linalg.norm(self.Kyy, 'fro')
                alignment.append(num / den)
            return alignment
        else:
            raise Exception("Invalid mode passed as argument.")



    def _NKD(self):
        """
        Calculate Norm of Kernel difference between combined kernel and target kernel.
        """

        return np.linalg.norm(self.combined_kernel - self.Kyy ,'fro')**2



    def NKD(self, X = None, y = None, mode = 'combo'):
        """
        Norm of Kernel Difference Calculation.

        Parameters
        ----------
        mode : str
            'combo' for computing NKD of combined kernel with target kernel
            'base' for computing NKD of each base kernel with target kernel

        X : array, shape = (n_samples_X, n_features)
            Feature vector for computing base kernels' NKD

        y : array, shape = (n_samples_X)
            Target variable for computing base kernels' NKD

        Returns
        -------
        align : array-like, shape = (n_kernels, )

        Raises
        ------
        Exception
            An exception is raised if an invalid mode is passed as argument
        """

        if mode is 'combo':
            return self._NKD()
        elif mode is 'base':
            if X is None or y is None:
                raise Exception('Error: Required data not supplied')
            self.make_base_kernels(X)
            self.make_target_kernel(y)
            return [np.linalg.norm(kernel - self.Kyy, 'fro') ** 2 for kernel in self.BaseKernels]
        else:
            raise Exception("Invalid mode passed as argument.")



    def HSIC(self, X = None, y = None, mode = 'combo', phase = 1):
        """
        Hilbert-Schmidt Independence Criterion Calculation.

        Parameters
        ----------
        mode : str
            'combo' for computing HSIC of combined kernel with target kernel
            'base' for computing HSIC of each base kernel with target kernel

        X : array, shape = (n_samples_X, n_features)
            Feature vector for computing base kernels' HSIC

        y : array, shape = (n_samples_X)
            Target variable for computing base kernels' HSIC

        Returns
        -------
        align : array-like, shape = (n_kernels, )

        Raises
        ------
        Exception
            An exception is raised if an invalid mode is passed as argument
        """

        if mode is 'combo':
            return self._HSIC()
        elif mode is 'base':
            if X is None or y is None:
                raise Exception('Error: Required data not supplied')
            if phase == 1:
                self.make_base_kernels(X)
            self.make_target_kernel(y)
            hsic = []
            N = len(X)
            ones = np.ones(shape = (N, 1)) / N
            return [np.sum(kernel * ones * self.Kyy * ones) / (N ** 2) for kernel in self.BaseKernels]
        else:
            raise Exception("Invalid mode passed as argument.")



    def _HSIC(self):
        """
        Calculate Hilbert-Schmidt Independence Criterion between combined kernel and target kernel.
        """

        N = self.combined_kernel.shape[0]
        ones = np.ones(shape = (N, 1)) / N
        return np.sum(self.combined_kernel * ones * self.Kyy * ones) / (N ** 2)



    def fit(self, X, y, phase):
        """
        Function that solves quadratic optimization problem based on similarity measure.

        Parameters
        ----------
        X : array, shape = (n_samples_X, n_features)
            Feature vector of training samples

        y : array, shape = (n_samples_X, )
            Target values for X

        Returns
        -------
        self : object
        """
        if phase == 1:
            self.make_base_kernels(X)
        self.make_target_kernel(y)
        self._get_optim_params()

        if self.similarity is 'KA' or self.similarity is 'NKD':
            if hasattr(self.Omega, "__len__"):
                P = matrix(2 * (self.__B + np.diag(self.Omega)))
            else:
                P = matrix(2 * (self.__B + self.Omega * np.eye(self.nKernels)))
            q = matrix(-self.__A)
        else:
            if hasattr(self.Omega, "__len__"):
                P = matrix(2 * np.diag(self.Omega))
            else:
                P = matrix(2 * self.Omega * np.eye(self.nKernels))
            q = matrix(-self.__C)
        G = matrix(-np.eye(self.nKernels))
        h = matrix(np.zeros(shape = (self.nKernels, )))
        res = solvers.qp(P, q, G, h)
        self.kernel_weights = np.abs(np.asarray(res['x']).flatten())
        self.kernel_weights /= np.sum(self.kernel_weights)
        return self



    def score(self, X, y, phase = 1):
        """
        Calculate similarity measure between combined kernel and target kernel.

        Parameters
        ----------
        X : array, shape = (n_samples_X, n_features)
            Feature vector of test samples

        y : array, shape = (n_samples_X, )
            True values for X

        Returns
        -------
        score : float
            similarity-score value

        Raises
        ------
        Exception
            An exception is raised if the function is called before self.fit()
        """

        if self.kernel_weights is None:
            raise Exception('Kernel combination weights have not been calculated yet.')
        self.make_target_kernel(y)
        self.make_combined_kernel(X, phase = phase)
        if self.similarity is 'KA':
            return self.KA(phase = phase)
        elif self.similarity is 'NKD':
            return -self.NKD()
        else:
            return self.HSIC()




class SMKL_SVR(BaseEstimator):
    """
    Class implementing SVR fitting with custom kernel built using SMKL class


    ...

    Attributes
    ----------
    C : float
        Regularization parameter of SVR

    epsilon : float
        Epsilon-tube with parameter of SVR

    kernel : function
        Base kernel type (rbf, polynomial)

    Omega : float or array-like, shape = (n_kernels, )
        Regularization parameter of SMKL

    similarity : str
        Similarity measure used in SMKL

    kernel_params : array-like, shape = (n_kernels, )
        Collection of base kernel parameters

    smkl : SMKL object
        SMKL object for learning base kernel weights

    ckernel : array, shape = (n_samples_X, n_samples_Y)
        Custom-built combined kernel

    svr : sklearn.svm.classes.SVR
        scikit-learn's implementation of SVR learning machine


    Methods
    -------
    fit(X, y)
        Standard method for fitting an estimator. Base kernel weights are first computed
        using SMKL object, the combined kernel is computed and finally an SVR object
        is trained.

    predict(X)
        Standard method for predicting target variables. Constructs combined kernel based on
        test feature vector and produces a vector of predictions.

    score(X, y)
        Standard method for evaluating score of estimator given a set of features
        and a target variable vector by calling the self.predict(X) method.
    """

    def __init__(self, C = 1, epsilon = 1e-1, kernel = rbf_kernel, Omega = 1, similarity = 'KA', kernel_params = None, bKernels = None, bweights = None):
        """
        Parameters
        ----------
        C : float
            Regularization parameter of SVR

        epsilon : float
            Epsilon-tube with parameter of SVR

        kernel : function
            Base kernel type (rbf, polynomial)

        Omega : float or array-like, shape = (n_kernels, )
            Regularization parameter of SMKL

        similarity : str
            Similarity measure used in SMKL

        kernel_params : array-like, shape = (n_kernels, )
            Collection of base kernel parameters

        smkl : SMKL object
            SMKL object for learning base kernel weights

        ckernel : array, shape = (n_samples_X, n_samples_Y)
            Custom-built combined kernel

        svr : sklearn.svm.classes.SVR
            scikit-learn's implementation of SVR learning machine
        """

        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.Omega = Omega
        self.similarity = similarity
        self.kernel_params = kernel_params
        self.BaseKernels = bKernels
        self.prevWeights = bweights
        self.phase = 1
        if self.BaseKernels is None:
            self.smkl = SMKL(Omega = self.Omega, kernel = self.kernel, similarity = self.similarity, kernel_params = self.kernel_params)
        else:
            self.smkl = SMKL(Omega = self.Omega, kernel = self.kernel, similarity = self.similarity, kernel_params = self.kernel_params, baseKernels = self.BaseKernels)
            self.phase = 2
        self.ckernel = None
        self.svr = None



    def fit(self, X, y):
        """
        Standard method for fitting an estimator. Base kernel weights are first computed
        using SMKL object, the combined kernel is computed and finally an SVR object
        is trained.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training feature vector

        y : array-like, shape = (n_samples)
            Vector of target variables

        Returns
        -------
        self : object
        """

        self.svr = SVR(C = self.C, epsilon = self.epsilon, kernel = 'precomputed')
        self._xtrain = X
        self.smkl.fit(X, y, self.phase)
        self.smkl.make_combined_kernel(X, phase = self.phase)
        ckernel = self.smkl.get_combined_kernel()
        self.svr.fit(ckernel, y)
        return self



    def predict(self, X):
        """
        Standard method for predicting target variables. Constructs combined kernel based on
        test feature vector and produces a vector of predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Testing feature vector

        Returns
        ------
        y_pred : array, shape = (n_samples,)
        """

        self._xtest = X
        self.smkl.make_combined_kernel(self._xtrain, self._xtest, phase = self.phase, Wsub = self.prevWeights)
        ckernel = self.smkl.get_combined_kernel()
        return self.svr.predict(ckernel.T)



    def score(self, X, y):
        """
        Standard method for evaluating score of estimator given a set of features
        and a target variable vector by calling the self.predict(X) method.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Feature vector used for prediction

        y : array-like, shape = (n_samples, n_features)
            Target variable vector against which the prediction output will be tested

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y
        """
        ypred = self.predict(X)
        #return r2_score(y, ypred)
        return mean_absolute_error([y],ypred)

    def stats(self,X,y):
        temp = self.predict(X)
        ypred = temp[0].astype(float)
        return ypred
        #ypred = self.predict(X)
        #rmse = math.sqrt(mean_squared_error(y,ypred))
        #CVrmse1 = 100*rmse/np.mean(ypred)
        #CVrmse2 = 100*rmse/np.mean(y)
        #mae = mean_absolute_error(y,ypred)
        #return rmse, CVrmse1, CVrmse2, mae
