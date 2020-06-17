import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)

	Note : Remember to normalize input data before processing further
	'''
	newarr=np.ones((X.shape[0], 1)).astype(float)
	for i in range(1,len(X[1])):
		if type(X[0,i])==str:
			labels=np.unique(X[:,i])
			demo=one_hot_encode(X[:,i],labels)
			newarr=np.append(newarr,demo,axis=1)
		else:
			mean=np.mean(X[:,i])
			std=np.std(X[:,i])
			demo=(X[:,i]-mean)/std
			newarr=np.append(newarr,demo.reshape(X.shape[0],1),axis=1).astype(float)
	return newarr, Y

def ordinary_least_squares(X, Y, lr=0.01):
	''' TASK 2

	X = input feature matrix [N X D]
	Y = output values [N X 1]
	Return the weight vector W, [D X 1] 
	'''
	m = len(Y)
	iterations=1000
	W = np.random.randn(X.shape[1],1)
	for it in range(iterations):
		prediction_error = np.matmul(X,W)-Y
		W = W -(1/m)*lr*( np.matmul(X.transpose(),prediction_error))
	return W

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 3
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	prediction_error = np.matmul(X,W)-Y
	return np.matmul(np.transpose(X),prediction_error) + (_lambda*W)

def ridge_grad_descent(X, Y, _lambda, max_iter=10000, lr=0.00001, epsilon = 1e-4):
	''' TASK 3 - PART A
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	m=len(Y)
	W=np.random.randn(X.shape[1],1).astype(float)
	for it in range(max_iter):
		W=W-lr*grad_ridge(W,X,Y,_lambda)
	return W


def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Lasso Loss Function 
	'''
	pass


if __name__ == "__main__":
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	W= ridge_grad_descent(trainX,trainY,100)
	print(np.linalg.norm(W))