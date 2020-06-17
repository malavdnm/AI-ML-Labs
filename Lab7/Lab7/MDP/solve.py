import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mdpfile')
parser.add_argument('--output')
args = parser.parse_args()
file = open(args.mdpfile, 'r')
lines = file.read().splitlines()

numStates , numActions, start , stop, discount = 0 , 0 , -1 , [], 1  
# print("here")
for line in lines:
	words = line.strip().split()
	if words[0] == 'numStates':
		numStates = int(words[1])
		# print(numStates)
	elif words[0] == 'numActions':
		numActions = int(words[1])
		# print(numActions)
	elif words[0] == 'start':
		#  initialize edges
		transitions = [[[] for i in range(numActions)] for j in range(numStates)]
		start = int(words[1])
		# print(start)
	elif words[0] == 'end':
		stop = list(map( int, words[1:]))
		# print(stop) # making the end list
	elif words[0] == 'transition':
		transitions[int(words[1])][int(words[2])].append([int(words[3]), float(words[4]) , float(words[5])])
	elif words[0] == 'discount':
		discount = float(words[1])
		# print(discount)
# print(transitions)
V = [ 0 for _ in range(numStates)]
action = [ 0 for _ in range(numStates)]

while True:
	newV=[ 0 for _ in range(numStates)]
	max_val = float("-inf")
	flag = True
	for i in range(numStates):
		if i in stop: continue
		max_val=-1000000
		for j in range(numActions):
			temp_value = None
			for next_State, reward, p in transitions[i][j]:
				if temp_value is None: temp_value = 0
				temp_value+=(reward+discount*V[next_State])*p
			if temp_value is not None and (temp_value>max_val):
				max_val=temp_value
				newV[i]=max_val
				action[i]=j
		if not np.isclose(newV[i],V[i],1e-10):
			flag = False
			V[i] = newV[i]
	if flag==True:
		break


sys.stdout = open(args.output, 'w')
for i in range(numStates):
	print(V[i],action[state],end="")