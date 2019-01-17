import numpy as np
from random import shuffle


def initializeMatrix(nWidth, occupRate, groupRatios):
	assert (occupRate >= 0 and occupRate <= 1),...
	"Occupancy Rate must be between 0 and 1."
	assert (sum(groupRatios) == 1), "Sum of weights must be 1."

	nGroups = len(groupRatios)
	nNodes = nWidth ** 2
	nodes = [0] * nNodes

	startNode = (1 - occupRate) * nNodes
	startNode = int(round(startNode))

	for i in range(nGroups):
		idG = i + 1
		nNodesThisGroup = groupRatios[i] * nNodes * occupRate
		nNodesThisGroup = int(round(nNodesThisGroup))
		endNode = startNode + nNodesThisGroup

		if i == (nGroups - 1) and endNode < nNodes:
			nodes[startNode:] = [idG] * len(nodes[startNode:])
		elif endNode > nNodes:
			nodes[startNode:] = [idG] * len(nodes[startNode:])
			break
		else:
			nodes[startNode: endNode] = [idG] * (endNode - startNode)

		startNode += nNodesThisGroup
		endNode += nNodesThisGroup

	shuffle(nodes)
	return np.reshape(nodes, (nWidth, nWidth))


def calcHappiness(initMap, weightMat, likeMat):
	assert (np.size(weightMat,0)%2 == 1 and np.size(weightMat,1)%2 == 1),...
	"Number of rows and columns of weight matrix should be both odd numbers."

	nRow = np.size(initMap,0)
	nCol = np.size(initMap,1)

	rVer = (np.size(weightMat,0) - 1) / 2
	rHor = (np.size(weightMat,1) - 1) / 2
	rVer = int(round(rVer))
	rHor = int(round(rHor))
	
	happMat = np.zeros_like(initMap, float)

	for i in range(nRow):
		for j in range(nCol):
			myGroup = initMap[i,j]
			# print(myGroup)
			if myGroup == 0:
				continue
			else:
				locMap = initMap[max(0,i-rVer) : min(nRow,i+rVer+1),:][:,max(0,j-rHor) : min(nCol,j+rHor+1)]
				wMt = np.copy(weightMat)
				if i < rVer:
					wMt = wMt[(rVer-i):,:]
				if (nRow - 1 - i) < rVer:
					wMt = wMt[:nRow-i+rVer,:]
				if j < rHor:
					wMt = wMt[:,(rHor-j):]
				if (nCol - 1 - j) < rHor:
					wMt = wMt[:,:nCol-j+rHor]
				
				locLikeMap = np.zeros_like(locMap, float)
				for k in range(np.size(locMap,0)):
					for l in range(np.size(locMap,1)):
						locLikeMap[k,l] = likeMat[myGroup][locMap[k,l]]

				locHappMap = locLikeMap * wMt
				happ = locHappMap.sum()

				maxLocHappMap = np.ones_like(locMap, float) * wMt
				maxHapp = maxLocHappMap.sum()

				happMat[i,j] = happ / maxHapp
	return happMat


nWidth = 10
occupRate = 0.8
groupRatios = [.5,.5]

# Weights of impact from neighbors, this person is in the center of te matrix.
# Rows and columns must be odd numbers.

# Example in class
weightMat = np.array([[1,1,1],
					  [1,0,1],
					  [1,1,1]])
# weightMat = np.array([[1,1,1,1,1],
# 					  [1,1,1,1,1],
# 					  [1,1,0,1,1],
# 					  [1,1,1,1,1],
# 					  [1,1,1,1,1]])

# Manhattan distance of 3, impact decreases by distance
# weightMat = np.array([[1,1,1],
# 					  [1,0,1],
# 					  [1,1,1]])


# Like Matrix is the likeliness between groups
# LikeMat[2][3] is how much group 2 likes group 3
# First row and column is the empty space
likeMat = [[0.0, 0.0, 0.0],
		   [0.0, 1.0, 0.3],
		   [0.0, 0.3, 1.0]]

# likeMat = [[1,0,0],
# 		   [0,1,0],
# 		   [0,0,1]]

t = 0.4

initMap = initializeMatrix(nWidth, occupRate, groupRatios)
happMat = calcHappiness(initMap, weightMat, likeMat)

print(initMap)
print(happMat)