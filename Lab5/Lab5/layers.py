import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes, activation):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		self.activation = activation
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		if self.activation == 'relu':
			# raise NotImplementedError
			self.data = relu_of_X(np.dot(X, self.weights) + self.biases)
			return self.data
		elif self.activation == 'softmax':
			# raise NotImplementedError
			self.data = softmax_of_X(np.dot(X, self.weights) + self.biases)
			return self.data
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		
		if self.activation == 'relu':
			inp_delta = gradient_relu_of_X(self.data, delta)
			inp_delta=inp_delta*delta
		elif self.activation == 'softmax':
			inp_delta = gradient_softmax_of_X(self.data, delta)
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		new_delta = np.dot(inp_delta, self.weights.transpose())
		self.weights -= lr*(np.dot((activation_prev.transpose()), inp_delta))
		self.biases -= lr*sum(inp_delta)
		return new_delta
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride, activation):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride
		self.activation = activation
		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.data = np.zeros((n , self.out_depth , self.out_row , self.out_col))
		for i in range(n):
			for d in range(self.out_depth):
				for x in range(0 , self.out_row):
					for y in range(0 , self.out_col):
						ax = x * self.stride 
						ay = y * self.stride
						Z = X[i, : , ax:(ax + self.filter_row) , ay:(ay + self.filter_col)]
						# print(Z.shape , d)
						self.data[i , d , x , y] = sum(sum(sum(Z * self.weights[d]))) +  self.biases[d]														
		# print(X.shape , Y.shape , self.stride)
		if self.activation == 'relu':
			return relu_of_X(self.data)
		elif self.activation == 'softmax':
			return softmax_of_X(self.data)

		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()

		
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		if self.activation == 'relu':
			delta = delta * gradient_relu_of_X(self.data,delta)
		elif self.activation == 'softmax':
			delta = delta * gradient_softmax_of_X(self.data)
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()

		new_Delta = np.zeros((n , self.in_depth , self.in_row , self.in_col))
		#  what should be the new delta ??

					# increase the weights for each instance of input box by adding input box * delta corresponding 
		# to do, first delta then update
		for i in range(n):
			for d in range(self.out_depth):
				for x in range(0 , self.out_row):
					for y in range(0 , self.out_col):
						ax = x * self.stride 
						ay = y * self.stride
						new_Delta[i , : , ax : (ax + self.filter_row ), ay:(ay + self.filter_col)] +=  delta[i, d, x , y] * self.weights[d, : , : , :] 
						self.weights[d] -=  lr * delta[i , d , x , y] * activation_prev[i , : , ax:(ax + self.filter_row) , ay:(ay + self.filter_col)] 
							# increase the weights for each instance of input box by adding input box * delta corresponding 
		for j in range(self.out_depth):
			self.biases[j] -=   lr * sum(sum(sum(delta[: , j , : , : ])))  
			# should be fine for biases
		return new_Delta
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		ans = np.zeros((n , self.out_depth , self.out_row , self.out_col))
		Y = np.zeros((n,self.out_depth,self.out_row,self.out_col))
		for i in range(n):
			for d in range(self.out_depth):
				for x in range(self.out_row):
					for y in range(self.out_col):
						ax = self.stride * x 
						ay = self.stride * y 
						Z = X[i , d , ax:(ax + self.filter_row) , ay:(ay + self.filter_col)]
						Y[i , d ,x , y] =  np.mean(Z)
		return Y
		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size
		new_Delta = np.zeros((n , self.in_depth , self.in_row , self.in_col))
		for i in range(n):
			for d in range(self.out_depth):
				for x in range(self.out_row):
					for y in range(self.out_col):
						ax = self.stride * x 
						ay = self.stride * y
						n=self.filter_row*self.filter_col 
						new_Delta[i , d , ax:ax+self.filter_row , ay:ay+self.filter_col] += delta[i,d,x,y]/n
		return new_Delta
		###############################################
		# TASK 2 - YOUR CODE HERE

		# raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Function for the activation and its derivative
def relu_of_X(X):

	# Input
	# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
	# Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation relu
	Y=X.copy()
	Y[X<0]=0
	# print(X)
	return Y
	# raise NotImplementedError
	
def gradient_relu_of_X(X, delta):
	# Input
	# data : Output from next layer/input | shape: batchSize x self.out_nodes
	# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
	# Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation relu amd during backwardpass
	# print(X,delta)
	# print(type(X),type(delta))
	Y=X.copy()
	Y[X<0]=0
	Y[X>0]=1
	return Y
	# raise NotImplementedError
	
def softmax_of_X(X):
	# Input
	# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
	# Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation softmax
	# print(X)
	# print(X[0])
	e_X=np.exp(X-np.max(X))
	s=np.diag(1/np.sum(e_X,1))
	return (s@e_X)
	# raise NotImplementedError
	
def gradient_softmax_of_X(X, delta):
	# Input
	# data : Output from next layer/input | shape: batchSize x self.out_nodes
	# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
	# Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation softmax amd during backwardpass
	# Hint: You might need to compute Jacobian first
	# print(X[0])
	ans=np.zeros((X.shape))
	# print(ans[])
	for i in range(len(X)):
		jacobian=X[i].reshape(X.shape[1],1)@X[i].reshape(1,X.shape[1])
		jacobian=np.diag(X[i])-jacobian
		ans[i]=delta[i]@jacobian
		# print(delta)
	# 	for j in range(len(mat)):
	# 		for k in range(len(mat)):
	# 			if j==k:
	# 				mat[j][k]=X[i][j]*(1-X[i][k])
	# 			else:
	# 				mat[j][k]=-X[i][j]*X[i][k]
	# 	# print("dfse",delta.dot(mat))
	# 	ans[i]=delta[i]@mat
	# # print(ans)
	return ans
	# raise NotImplementedError
	