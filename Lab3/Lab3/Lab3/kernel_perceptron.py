import numpy as np
from numpy import linalg

def linear_kernel(x1, x2):
    return x1@x2

def polynomial_kernel(x, y, degree=3):
    return (1+x@y)**3

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-0.5*linalg.norm(x-y)**2/sigma**2)

class KernelPerceptron(object):

    def __init__(self, kernel=linear_kernel, iterations=1):
        self.kernel = kernel
        self.iterations = iterations
        self.alpha = None
        self.X_train = None

    def fit(self, X, y):
        ''' find the alpha values here'''
        self.X_train=X.copy()
        n_samples, n_features = X.shape
        self.alpha = np.zeros((n_samples), dtype=np.float64)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            # print(i)
            for j in range(i,n_samples):
                # print(j)
                K[i,j] = self.kernel(X[i], X[j])
                K[j,i] = K[i,j]
        for t in range(self.iterations+5):
            # print(t)
            for i in range(n_samples):
                # a=np.multiply(y,self.alpha)
                # print(a,a.shape,type(a))
                # a=K[i].dot(a)
                # print(a,a.shape,type(a))
                # b=K[:,i].dot(np.multiply(self.alpha, y))
                # print(b.shape,type(b))
                if np.sign(K[:,i].dot(self.alpha)) != y[i]:
                    self.alpha[i] += y[i]
        print(self.alpha)
    
    def project(self, X):
        '''return projected values from alpha and corresponding support vectors'''
        y_predict = np.zeros(len(X))
        n_samples, n_features = X.shape
        # print("here")
        for i in range(len(self.X_train)):
            for j in range(n_samples):
                # print(,self.kernel(self.X_train[i], X[j]) * self.alpha[i])
                y_predict[j]+=self.kernel(self.X_train[i], X[j]) * self.alpha[i]
        return y_predict
        

    def predict(self, X):
        X = np.atleast_2d(X)
        # print(X.shape)
        return np.sign(self.project(X))
