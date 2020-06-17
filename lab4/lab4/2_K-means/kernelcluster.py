from copy import deepcopy
from itertools import cycle
from pprint import pprint as pprint
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
import math



def RBFKernel(p1,p2,sigma=3):
	'''
	p1: tuple: 1st point
	p2: tuple: 2nd point
	Returns the value of RBF kernel
	'''
	value = np.exp(0.5*np.linalg.norm(p1-p2)/sigma**2)
	# TODO [task3]
	# Your function must work for all sized tuples.
	return value

def initializationrandom(data,C,seed=45):
	'''
	data: list of tuples: the list of data points
	C:int : number of cluster centroids
    seed:int : seed value for random value generator
	Returns a list of tuples,representing the cluster centroids and a list of list of tuples representing the cluster  
	'''
	# centroidList =  []
	# clusterLists = [[] for i in range(C)]
	# TODO [task3]:
	np.random.seed(seed)
	np.random.shuffle(data)
	clusterLists=np.array_split(data,C)
	centroidList=np.array(tuple(map((lambda y: np.mean(y,axis=0)), clusterLists)))
	# Initialize the cluster centroids by sampling k unique datapoints from data and assigning a data point to a random cluster
	assert len(centroidList) == C
	assert len(clusterLists) == C
	return centroidList,clusterLists

def firstTerm(p):
	'''
	p: a tuple for a  datapoint
	'''
	# value = None
	'''
	# TODO [task3]:
	# compute the first term in the summation of distance.
	'''
	value = RBFKernel(p, p)
	return value

def secondTerm(p,pi_k,sigma=3):
	'''
	data : list of tuples: the list of data points
	pi_k : list of tuples: the list of data points in kth cluster
	'''
	value = np.mean(np.exp((-0.5/sigma**2)*np.sum((pi_k-p)**2,1)))
	'''
	# TODO [task3]:
	# compute the second term in the summation of distance.
	'''

	return -2*value

def thirdTerm(pi_k,sigma=3):
	'''
	pi_k : list of tuples: the list of data points in kth cluster
	'''
	value = None
	'''
	# TODO [task3]:
	# compute the third term in the summation of distance.
	'''
	# print(len(pi_k))
	np_pi_k = np.array(pi_k)
	N=np.sum(np_pi_k**2,1) #norm
	m=N.shape[0]
	temp=np_pi_k@(np_pi_k.T) #phi ij
	value=N+N.reshape((m,1))-2*temp
	value=np.sum(np.exp(-value/(2*sigma**2)))
	value=value/m**2
	return value

def hasconverged(prevclusterList,clusterList,C):
	'''
	prevclusterList : list of (list of tuples): the list of lists of  tuples of datapoints in a cluster in previous iteration
	clusterList: list of (list of tuples): the list of lists of tuples of datapoints in a cluster
	C: int : number of clusters
	'''
	converged = False
	'''
	# TODO [task3]:
	check if the cluster membership of the clusters has changed or not.If not,return True. 
	'''
	converged = False

	# TODO [task1]:
	# Use Euclidean distance to measure centroid displacements.
	for i in range(len(prevclusterList)):
		if len(prevclusterList[i])==len(clusterList[i]):
			for i in range(prevclusterList):
				if (prevclusterList[i]==clusterList[i]).all():
					pass
				else:
					return converged
		else:
			return converged
	converged=True
	########################################
	return converged 
	
def kernelkmeans(data,C,maxiter=10):
	'''
	data : list of tuples: the list of data points
	C: int : number of clusters
	'''
	# print(type(data))
	# return
	centroidList,clusterLists = initializationrandom(data,C)
	'''
	# TODO [task3]:
	# iteratively modify the cluster centroids.
	# Stop only if convergence is reached, or if max iterations have been exhausted.
	# Save the results of each iteration in all_centroids.
	# Tip: use deepcopy() if you run into weirdness.
	'''
	for _ in range(maxiter):
		print(_)
		third=[]
		for i in range(C):
			third.append(thirdTerm(clusterLists[i]))
		newClusterLists = [[] for i in range(C)]
		for i in range(len(data)):
			distance=[]
			for j in range(C):
				distance.append(secondTerm(data[i],clusterLists[j])+third[j])
			indx=min(enumerate(distance), key=lambda x: x[1])[0]
			newClusterLists[indx].append(data[i])
		if hasconverged(newClusterLists,clusterLists,C):
			break
		clusterLists=np.array(deepcopy(newClusterLists))
	centroidList=np.array(tuple(map((lambda y: np.mean(y,axis=0)), clusterLists)))
	return clusterLists,centroidList




def plot(clusterLists,centroidList,C):
	color = iter(cm.rainbow(np.linspace(0,1,C)))
	plt.figure("result")
	plt.clf()
	for i in range(C):
		col = next(color)
		memberCluster = np.asmatrix(clusterLists[i])
		plt.scatter(np.ravel(memberCluster[:,0]),np.ravel(memberCluster[:,1]),marker=".",s =100,c = col)
	color = iter(cm.rainbow(np.linspace(0,1,C)))
	for i in range(C):
		col = next(color)
		plt.scatter(np.ravel(centroid[i,0]),np.ravel(centroid[i,1]),marker="*",s=400,c=col,edgecolors="black")
	plt.show()


filePath1 = "datasets/mouse.csv"
filePath2 = "datasets/3lines.csv"
mouse  = np.loadtxt(open(filePath1, "rb"), delimiter=",", skiprows=1)
lines3 = np.loadtxt(open(filePath2, "rb"), delimiter=",", skiprows=1)
clusterResult, centroid = kernelkmeans(mouse,C=3)
plot(clusterResult, centroid,C=3)
clusterResult,centroid = kernelkmeans(lines3,C=3)
plot(clusterResult, centroid,C=3)
#save the plots accordingly

