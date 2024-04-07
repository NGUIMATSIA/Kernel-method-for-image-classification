import numpy as np
import cvxopt
from cvxopt import matrix


class KernelSVC:

    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None

    def fit(self, X, y):
        N = len(y)
        K = self.kernel(X, X)

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(N))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.hstack((np.zeros(N), np.ones(N) * self.C)))
        A = matrix(y * np.ones((1, N)))
        b = matrix(0.0)
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x']).flatten()

        self.sv_index = (self.alpha > self.epsilon)
        self.support = X[self.sv_index]
        alpha_sv = self.alpha[self.sv_index]
        y_diag_sv = np.diag(y[self.sv_index])

        self.b = (y[self.sv_index] - alpha_sv.T @ y_diag_sv @ self.kernel(self.support, self.support)).mean()
        self.norm_f = alpha_sv.T @ y_diag_sv @ self.kernel(self.support, self.support) @ y_diag_sv @ alpha_sv
        self.part_f = alpha_sv.T @ y_diag_sv


    ### Implementation of the separting function $f$
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.part_f @ self.kernel(self.support, x)


    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1

    
class Multi_Class_SVM_Classifier_OvA(object):

    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.classifiers = []
        

    def fit(self, X_train, y_train):

        self.nclasses = np.unique(y_train).size
        labels = np.unique(y_train)

        
        for i in range(self.nclasses):

            svm = KernelSVC(C = self.C, kernel = self.kernel)
            y_tr = np.where(y_train == labels[i], 1, -1)
            svm.fit(X_train, y_tr)
            self.classifiers.append(svm)

    def predict(self, X_test):
        predicts = np.zeros((X_test.shape[0], self.nclasses))

        for count, classifier in enumerate(self.classifiers):

            predicts[:,count] = classifier.separating_function(X_test) + classifier.b

        return np.argmax(predicts, axis = 1)
    

class Multi_Class_SVM_Classifier_OvO(object):

    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.classifiers = []

    def fit(self, X_train, y_train):

        self.nclasses = np.unique(y_train).size
        labels = np.unique(y_train)

        
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):

                svm = KernelSVC(C = self.C, kernel = self.kernel)

                # keep only labels i and j for binary classification
                indexes = np.logical_or(y_train == labels[i],y_train == labels[j])
                y_tr = np.where(y_train[indexes] == labels[i],1,-1)

                svm.fit(X_train[indexes], y_tr)
                self.classifiers.append([svm,labels[i],labels[j]])
    def predict(self, X_test):
        predicts = np.zeros((X_test.shape[0], self.nclasses))

        for [classifier,label1, label2] in self.classifiers:

            pred = classifier.predict(X_test)
            predicts[np.where(pred == 1),label1] +=1
            predicts[np.where(pred == -1),label2] +=1

        return np.argmax(predicts, axis = 1)