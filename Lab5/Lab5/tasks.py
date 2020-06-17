import nn
import numpy as np
import sys

from util import *
from visualize import *
from layers import *


# XTrain - List of training input Data
# YTrain - Corresponding list of training data labels
# XVal - List of validation input Data
# YVal - Corresponding list of validation data labels
# XTest - List of testing input Data
# YTest - Corresponding list of testing data labels

def taskSquare(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSquare()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.1 - YOUR CODE HERE
	out_nodes = 2
	alpha = 0.01
	batchSize = 32
	epochs = 20
	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(FullyConnectedLayer(2, 4,'relu'))
	nn1.addLayer(FullyConnectedLayer(4, 2,'softmax'))
	# raise NotImplementedError
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 2'
	# Use drawSquare(XTest, pred) to visualize YOUR predictions.
	if draw:
		drawSquare(XTest, pred)
	return nn1, XTest, YTest


def taskSemiCircle(draw):
	XTrain, YTrain, XVal, YVal, XTest, YTest = readSemiCircle()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.2 - YOUR CODE HERE
	# raise NotImplementedError
	out_nodes = 2
	alpha = 0.03
	batchSize = 16
	epochs = 25
	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(FullyConnectedLayer(2, 4,'relu'))
	nn1.addLayer(FullyConnectedLayer(4, 2,'softmax'))
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	# Run script visualizeTruth.py to visualize ground truth. Run command 'python3 visualizeTruth.py 4'
	# Use drawSemiCircle(XTest, pred) to vnisualize YOUR predictions.
	if draw:
		drawSemiCircle(XTest, pred)
	return nn1, XTest, YTest

def taskMnist():
	XTrain, YTrain, XVal, YVal, XTest, YTest = readMNIST()
	# Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# Add layers to neural network corresponding to inputs and outputs of given data
	# Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	###############################################
	# TASK 2.3 - YOUR CODE HERE
	# raise NotImplementedError
	out_nodes = 10
	alpha = 0.003
	batchSize = 32
	epochs = 20
	nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)
	nn1.addLayer(FullyConnectedLayer(784,16,'relu'))
	nn1.addLayer(FullyConnectedLayer(16, 10,'softmax'))	
	###############################################
	nn1.train(XTrain, YTrain, XVal, YVal, False, True)
	pred, acc  = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)
	return nn1, XTest, YTest

def taskCifar10():	
	XTrain, YTrain, XVal, YVal, XTest, YTest = readCIFAR10()
	idx = np.random.choice(40000,1500)
	XTrain = XTrain[idx,:,:,:]
	XVal = XVal[0:1000,:,:,:]
	XTest = XTest[0:1000,:,:,:]
	YVal = YVal[0:1000,:]
	YTest = YTest[0:1000,:]
	YTrain = YTrain[idx,:]
	np.random.seed(42)
	
	modelName = 'model.npy'
	# # Create a NeuralNetwork object 'nn1' as follows with optimal parameters. For parameter definition, refer to nn.py file.
	# # nn1 = nn.NeuralNetwork(out_nodes, alpha, batchSize, epochs)	
	# # Add layers to neural network corresponding to inputs and outputs of given data
	# # Eg. nn1.addLayer(FullyConnectedLayer(x,y))
	# ###############################################
	# # TASK 2.4 - YOUR CODE HERE
	nn1 = nn.NeuralNetwork(10,0.01,8,24)
	nn1.addLayer(ConvolutionLayer([3,32,32], [4,4], 8, 2,'relu'))
	nn1.addLayer(AvgPoolingLayer([8,15,15], [3,3], 2))
	nn1.addLayer(FlattenLayer())
	nn1.addLayer(FullyConnectedLayer(7*7*8,20,'relu'))
	nn1.addLayer(FullyConnectedLayer(20,10,'softmax'))

	###################################################
	return nn1,  XTest, YTest, modelName # UNCOMMENT THIS LINE WHILE SUBMISSION

	print("HERE")
	nn1.train(XTrain, YTrain, XVal, YVal, True, True, loadModel=True, saveModel=True, modelName=modelName)
	pred, acc = nn1.validate(XTest, YTest)
	print('Test Accuracy ',acc)