import numpy as np
from random import shuffle


def initializeMatrix(nWidth, occupRate, groupRatios):
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



t = 3

effRadius = 1

nWidth = 10
occupRate = 0.9
groupRatios = [.4, .4, .2]

x = initializeMatrix(nWidth, occupRate, groupRatios)
print(x)
