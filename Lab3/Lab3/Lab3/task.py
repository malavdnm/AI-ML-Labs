import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 1
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	newarr=np.ones((X.shape[0], 1))
	for i in range(1,len(X[1])):
		if type(X[0,i])==str:
			labels=np.unique(X[:,i])
			demo=one_hot_encode(X[:,i],labels)
			newarr=np.append(newarr,demo,axis=1)
		else:
			mean=np.mean(X[:,i])
			std=np.std(X[:,i])
			demo=(X[:,i]-mean)/std
			newarr=np.append(newarr,demo.reshape(X.shape[0],1),axis=1)
	Y=np.where(Y=='yes',1,0)
	# Y.eq('no').mul(0)
	return newarr.astype(float), Y.astype(float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0*x))

def logistic_train(X, Y, lr=0.01, max_iter = 500):
	''' TASK 1
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	lr 			= learning rate
	max_iter 	= maximum number of iterations of gradient descent to run
	Return the trained weight vector [D X 1] after performing gradient descent
	'''
	m = len(Y)
    # cost_history = np.zeros((iterations,1))
	W= np.zeros((X.shape[1],1))
	for i in range(max_iter):
		W = W - (lr/m) * (X.T @ (sigmoid(X @ W) -Y)) 
        # cost_history[i] = compute_cost(X, y, params)
	
	return W
	

def logistic_predict(X, Weights):
	''' TASK 1
	X 			= input feature matrix [N X D]
	Weights		= weight vector
	Return the predictions as [N X 1] vector
	'''
	return np.round(sigmoid(X @ Weights))
