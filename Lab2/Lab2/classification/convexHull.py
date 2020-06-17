import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
random.seed(42)


def visualize(points):
	''' Write the code here '''
	allpoints=np.array(points,dtype=object)
	# print(allpoints[0][0][0])
	# a['A']=[[]]
	# a['B']=[[]]
	# check_for_A=['A1','A2','A3']
	# for i in range(len(allpoints)):
	# 	# print(allpoints[i][0])
	# 	if check_for_A.count(allpoints[i][0]):
	# 		if a.get(A):
	# 		a['A']=np.append(a['A'],np.array(allpoints[i][1:]).reshape(1,2).astype(float),axis=0)
	# 	else:
	# 		a['B']=np.append(a['B'],np.array(allpoints[i][1:]).reshape(1,2).astype(float),axis=0)

	a={}
	for i in range(len(allpoints)):
		if not(allpoints[i][0][0] in a):
			a[allpoints[i][0][0]]=np.array(allpoints[i][1:]).reshape(1,2).astype(float)
		else:
			a[allpoints[i][0][0]]=np.append(a[allpoints[i][0][0]],np.array(allpoints[i][1:]).reshape(1,2).astype(float),axis=0)
	# print(a)
	for values in a.values():
		hull=ConvexHull(values)
		plt.plot(values[:,0], values[:,1], 'o')
		for simplex in hull.simplices:
			plt.plot(values[simplex, 0], values[simplex, 1], 'k-')
	plt.show()

def grade():
	A,B = [],[]
	for i in range(3):
		A.append('A'+str(i))
		B.append('B'+str(i))

	points  = [3,5,7]
	allpoints = []
	till = 3
	for i in range(till):
		coords = np.array([(A[i], random.random()*(100.0/points[i]), random.random()*(100.0)) for _ in range(points[i])])
		coords1 = np.array([(B[i], 25 + random.random()*(100.0/points[i]), random.random()*(100.0)) for _ in range(points[i])]) 
		allpoints.extend(coords)
		allpoints.extend(coords1)

	random.shuffle(allpoints)
	visualize(allpoints)
	return allpoints

a=grade()

