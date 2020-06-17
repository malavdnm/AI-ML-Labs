import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--mdpfile')
args = parser.parse_args()

gridfile = open(args.input, 'r')
lines = gridfile.read().splitlines()
gridfile.close()

grid = np.array([line.split() for line in lines], dtype=np.int64)
numStates = 0
start = -1
end = -1
numActions = 4
discount = 0.95
row, col = grid.shape
reward = -1
finalreward = 100
state = np.array([[-1]*row]*col, dtype=np.int64)
transitions = []
p=1

for i in range(row):
    for j in range(col):
        if grid[i][j] == 0:
            pass
        elif grid[i][j] == 1:
            state[i][j] = -1
            continue
        elif grid[i][j] == 2:
            start = numStates
        elif grid[i][j] == 3:
            end = numStates
        else:
            print("Error: Unknown character")
        
        state[i][j] = numStates
        numStates += 1
 
for i in range(row):
    for j in range(col):
        if state[i][j]!=-1:
            if i > 0:
                if ((state[i-1][j] != -1) and (state[i-1][j]) == end):
                    transitions.append(("transition", state[i, j], 0, state[i-1, j] , finalreward, p))
                elif (state[i-1][j] != -1):
                    transitions.append(("transition", state[i, j], 0, state[i-1, j] , reward, p))
            if j < col-1:
                if ((state[i][j+1] != -1) and (state[i][j+1]) == end):
                    transitions.append(("transition", state[i, j], 1, state[i, j+1] , finalreward, p))
                elif (state[i][j+1] != -1):
                    transitions.append(("transition", state[i, j], 1, state[i, j+1] , reward, p))
            if i < row-1:
                if ((state[i+1][j] != -1) and (state[i+1][j]) == end):
                    transitions.append(("transition", state[i, j], 2, state[i+1, j] , finalreward, p))
                elif (state[i+1][j] != -1):
                    transitions.append(("transition", state[i, j], 2, state[i+1, j] , reward, p))
            if j > 0:
                if ((state[i][j-1] != -1) and (state[i][j-1]) == end):
                    transitions.append(("transition", state[i, j], 3, state[i, j-1] , finalreward, p))
                elif (state[i][j-1] != -1):
                    transitions.append(("transition", state[i, j], 3, state[i, j-1] , reward, p))

sys.stdout = open(args.mdpfile, 'w')
print("numStates", numStates)
print("numActions", numActions)
print("start", start)
print("end", end)
for move in transitions:
        print(move[0],move[1],move[2],move[3],move[4],move[5])
    # print("\n")
print("discount", discount, end="")