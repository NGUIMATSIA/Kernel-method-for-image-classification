import numpy as np
import cvxopt
from cvxopt import matrix

class KernelSVC:
    def __init__(self, C, kernel, epsilon=1e-3):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None

    def fit(self, X1, X2, y, hog = True):

        if hog:
            X = X1
        else:
            X = X2
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

    def separating_function(self, x1, x2, hog = True):
        if hog:
            x = x1
        else:
            x = x2
        return self.part_f @ self.kernel(self.support, x)

    def predict(self, X):
        d = self.separating_function(X)
        return np.sign(d + self.b).astype(int)


    
class Ensemble_SVM_Classifier_OvA(object):

    def __init__(self, C, gaussian, Chi2, gaussian_Plus_Chi2, gaussian_prod_Chi2):
        self.C = C
        self.gaussian = gaussian
        self.Chi2 = Chi2
        self.gaussian_plus_Chi2 = gaussian_Plus_Chi2
        self.gaussian_prod_Chi2 = gaussian_prod_Chi2
        self.classifiers_hog = []
        self.classifiers_sift = []


    def fit(self, hog_feature, sift_feature, y_train):
        self.nclasses = np.unique(y_train).size
        labels = np.unique(y_train)
        print(self.nclasses)

        
        # Gaussian kernel and hog feature
        #for i in range(self.nclasses):
        #    svm = KernelSVC(C = self.C, kernel = self.gaussian)
        #    y_tr = np.where(y_train == labels[i], 1, -1)
        #   svm.fit(hog_feature, sift_feature, y_tr)
         #   self.classifiers_hog.append(svm)

        # Chi2 kernel and hog feature
        #for i in range(self.nclasses):
        #    svm = KernelSVC(C = self.C, kernel = self.Chi2)
        #    y_tr = np.where(y_train == labels[i], 1, -1)
        #    svm.fit(hog_feature, sift_feature, y_tr)
        #    self.classifiers_hog.append(svm)

        # Gaussian plus chi2 kernel and hog feature
        #for i in range(self.nclasses):
        #    svm = KernelSVC(C = self.C, kernel = self.gaussian_plus_Chi2)
        #    y_tr = np.where(y_train == labels[i], 1, -1)
        #    svm.fit(hog_feature, sift_feature, y_tr)
        #    self.classifiers_hog.append(svm)

        # Gaussian prod chi2 kernel and hog feature
        #for i in range(self.nclasses):
        #    svm = KernelSVC(C = self.C, kernel = self.gaussian_prod_Chi2)
        #    y_tr = np.where(y_train == labels[i], 1, -1)
        #    svm.fit(hog_feature, sift_feature, y_tr)
        #    self.classifiers_hog.append(svm)

        # chi2 kernel and sift feature
        for i in range(self.nclasses):
            svm = KernelSVC(C = self.C, kernel = self.Chi2)
            y_tr = np.where(y_train == labels[i], 1, -1)
            svm.fit(hog_feature, sift_feature, y_tr, hog = False)
            self.classifiers_sift.append(svm)

    def predict(self, hog_feature, sift_feature):
        predicts = np.zeros((hog_feature.shape[0], self.nclasses))

        #for count, classifier in enumerate(self.classifiers_hog):
        #    predicts[:,count%10] += classifier.separating_function(hog_feature, sift_feature) + classifier.b

        for count, classifier in enumerate(self.classifiers_sift):
            predicts[:,count%10] += classifier.separating_function(hog_feature, sift_feature, hog=False) + classifier.b

        return np.argmax(predicts, axis = 1)
    

class Ensemble_SVM_OvO(object):

    def __init__(self, C, gaussian, Chi2, gaussian_Plus_Chi2, gaussian_prod_Chi2):
        self.C = C
        self.gaussian = gaussian
        self.Chi2 = Chi2
        self.gaussian_plus_Chi2 = gaussian_Plus_Chi2
        self.gaussian_prod_Chi2 = gaussian_prod_Chi2
        self.classifiers_gauss_hog = []
        self.classifiers_chi2_hog = []
        self.classifiers_gauss_pl_chi2_hog = []
        self.classifiers_gauss_pr_chi2_hog  = []
        self.classifiers_chi2_sift = []


    def fit(self, hog_feature, sift_feature, y_train):

        self.nclasses = np.unique(y_train).size
        labels = np.unique(y_train)

        # Gaussian kernel plus hog feature
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):

                svm = KernelSVC(C = self.C, kernel= self.gaussian)

                # keep only labels i and j for binary classification
                indexes = np.logical_or(y_train == labels[i],y_train == labels[j])
                y_tr = np.where(y_train[indexes] == labels[i],1,-1)

                svm.fit(hog_feature[indexes], y_tr)
                self.classifiers_gauss_hog.append([svm,labels[i],labels[j]])

        # Chi2 kernel plus hog feature
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):

                svm = KernelSVC(C = self.C, kernel= self.Chi2)

                # keep only labels i and j for binary classification
                indexes = np.logical_or(y_train == labels[i],y_train == labels[j])
                y_tr = np.where(y_train[indexes] == labels[i],1,-1)

                svm.fit(hog_feature[indexes], y_tr)
                self.classifiers_gauss_hog.append([svm,labels[i],labels[j]])

        # Chi2 sum gaussian kernel  plus hog feature
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):

                svm = KernelSVC(C = self.C, kernel= self.gaussian_plus_Chi2)

                # keep only labels i and j for binary classification
                indexes = np.logical_or(y_train == labels[i],y_train == labels[j])
                y_tr = np.where(y_train[indexes] == labels[i],1,-1)

                svm.fit(hog_feature[indexes], y_tr)
                self.classifiers_gauss_hog.append([svm,labels[i],labels[j]])
        
        # Chi2 prod gaussian kernel  plus hog feature
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):

                svm = KernelSVC(C = self.C, kernel= self.gaussian_prod_Chi2)

                # keep only labels i and j for binary classification
                indexes = np.logical_or(y_train == labels[i],y_train == labels[j])
                y_tr = np.where(y_train[indexes] == labels[i],1,-1)

                svm.fit(hog_feature[indexes], y_tr)
                self.classifiers_gauss_hog.append([svm,labels[i],labels[j]])
        
        # Chi2 kernel  plus sift feature
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):

                svm = KernelSVC(C = self.C, kernel= self.Chi2)

                # keep only labels i and j for binary classification
                indexes = np.logical_or(y_train == labels[i],y_train == labels[j])
                y_tr = np.where(y_train[indexes] == labels[i],1,-1)

                svm.fit(sift_feature[indexes], y_tr)
                self.classifiers_gauss_hog.append([svm,labels[i],labels[j]])

        # Chi2  kernel  plus sift feature
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):

                svm = KernelSVC(C = self.C, kernel= self.gaussian_prod_Chi2)

                # keep only labels i and j for binary classification
                indexes = np.logical_or(y_train == labels[i],y_train == labels[j])
                y_tr = np.where(y_train[indexes] == labels[i],1,-1)

                svm.fit(hog_feature[indexes], y_tr)
                self.classifiers_gauss_hog.append([svm,labels[i],labels[j]])


    def predict(self, X_test1, X_test2):
        predicts = np.zeros((X_test1.shape[0], self.nclasses))

        for [classifier,label1, label2] in self.classifiers:

            pred = classifier.predict(X_test1, X_test2)
            predicts[np.where(pred == 1),label1] +=1
            predicts[np.where(pred == -1),label2] +=1

        return np.argmax(predicts, axis = 1)