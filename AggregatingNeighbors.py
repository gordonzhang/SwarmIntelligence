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


nWidth = 10
occupRate = 0.9
groupRatios = [.5, .5]
likeMat = []

initMap = initializeMatrix(nWidth, occupRate, groupRatios)
print(initMap)


def calcHappiness(initMap, weightMat, likeMat):
	assert (np.size(weightMat,0)%2 == 1 and np.size(weightMat,1)%2 == 1),...
	"Number of rows and columns of weight matrix should be both odd numbers."

	nRow = np.size(initMap,0)
	nCol = np.size(initMap,1)

	rVer = (np.size(weightMat,0) - 1) / 2
	rHor = (np.size(weightMat,1) - 1) / 2
	rVer = int(round(rVer))
	rHor = int(round(rHor))
	
	happMt = np.zeros_like(initMap)

	for i in range(nRow):
		for j in range(nCol):
			myGroup = initMap[i,j]
			locMap = initMap[max(0,i-rVer) : min(nRow,i+rVer+1),:][:,max(0,j-rHor) : min(nCol,j+rHor+1)]
			wMt = np.copy(weightMat)

			if i < rVer:
				wMt = wMt[(rVer-i):,:]
				# print(wMt)
			if (nRow - 1 - i) < rVer:
				wMt = wMt[:nRow-i+rVer,:]
				# print(wMt)
			if j < rHor:
				wMt = wMt[:,(rHor-j):]
				# print(wMt)
			if (nCol - 1 - j) < rHor:
				wMt = wMt[:,:nCol-j+rHor]
			happiness = locMap * wMt
			print(locMap)
			print(wMt)
			print(myGroup)
			print()



weightMat = np.array([[1,1,1],
					  [1,0,1],
					  [1,1,1]])

weightMat = np.random.rand(5,5)
print(weightMat)
calcHappiness(initMap, weightMat)