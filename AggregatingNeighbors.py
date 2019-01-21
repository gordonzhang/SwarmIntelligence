import numpy as np
from random import shuffle
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors


## Functions
def InitializeMatrix(nWidth:int, occupancyRate:float, gRatios:list):
	assert (0 <= occupancyRate <= 1), ...
	"Occupancy Rate must be between 0 and 1."
	assert (sum(gRatios) == 1), "Sum of weights must be 1."

	nGroups: int = len(gRatios)
	nNodes = nWidth ** 2
	nodes = [0] * nNodes

	startNode = (1 - occupancyRate) * nNodes
	startNode = int(round(startNode))

	for i in range(nGroups):
		idG = i + 1
		nNodesThisGroup = gRatios[i] * nNodes * occupancyRate
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


def CalcHappiness(aggMap:np.ndarray, weightMat:np.ndarray, likeMat:np.ndarray, threshold:float):
	assert (np.size(weightMat,0)%2 == 1 and np.size(weightMat,1)%2 == 1), ...
	"Number of rows and columns of weight matrix should be both odd numbers."

	nRow = np.size(aggMap,0)
	nCol = np.size(aggMap,1)

	rVer = (np.size(weightMat,0) - 1) / 2
	rHor = (np.size(weightMat,1) - 1) / 2
	rVer = int(round(rVer))
	rHor = int(round(rHor))
	
	happinessMat = np.zeros_like(aggMap, float)
	vacancyList = []
	unHappyList = []

	for i in range(nRow):
		for j in range(nCol):
			myGroup = aggMap[i,j]
			# print(myGroup)
			if myGroup == 0:
				vacancyList.append([i,j])
			else:
				locMap = aggMap[max(0,i-rVer) : min(nRow,i+rVer+1),:][:,max(0,j-rHor) : min(nCol,j+rHor+1)]
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
				happVal = locHappMap.sum()

				# Maximum happiness is calculated with all 1's in local like map
				# Happiness is the proportion of max happiness.
				maxLocHappinessMap = np.ones_like(locMap, float) * wMt
				maxHappiness = maxLocHappinessMap.sum()
				# Two options to calculate happiness:
				#   1. absolute value
				#   2. The ratio of absolute value and maximum possible value.
				# An interesting difference is that, in option 2, no one want to live on border.
				# happinessScale = happVal / maxHappiness
				happinessScale = happVal
				happinessMat[i,j] = happinessScale
				if happinessScale < threshold:
					unHappyList.append([i,j])

	return happinessMat, unHappyList, vacancyList


def move(aggMap:np.ndarray, unHappyList:list, vacancyList:list, movingSize:int):
	moveOutList = []
	moveInList = []
	actualMoveSize = min(len(unHappyList), len(vacancyList), movingSize)
	moveOutIdxList = np.random.choice(len(unHappyList), actualMoveSize, replace=False)
	for i in moveOutIdxList:
		moveOutList.append(unHappyList[i])

	moveInIdxList = np.random.choice(len(vacancyList), actualMoveSize, replace=False)
	for i in moveInIdxList:
		moveInList.append(vacancyList[i])

	for outIdx, inIdx in zip(moveOutList, moveInList):
		aggMap[outIdx[0],outIdx[1]], aggMap[inIdx[0],inIdx[1]] = aggMap[inIdx[0],inIdx[1]], aggMap[outIdx[0],outIdx[1]]
	return aggMap


def HappinessListFromMatrix(hMat:np.ndarray):
	hList = hMat.reshape(-1)
	index = np.argwhere(hList == 0)
	hList = np.delete(hList, index)
	return hList


if "__main__" == __name__:
	# Weights of impact from neighbors, this person is in the center of te matrix.
	# Rows and columns must be odd numbers.

	# Example in class
	# weightMat = np.array([[1,1,1],
	# 					  [1,0,1],
	# 					  [1,1,1]])

	# Manhattan distance of 2, impact decreases by distance
	weightMatrix =np.array([[0.0, 0.0, 0.5, 0.0, 0.0],
							[0.0, 0.5, 1.0, 0.5, 0.0],
							[0.5, 1.0, 0.0, 1.0, 0.5],
							[0.0, 0.5, 1.0, 0.5, 0.0],
							[0.0, 0.0, 0.5, 0.0, 0.0]])
	# Normalize weight matrix
	weightMatrix = weightMatrix / weightMatrix.sum()
	print("Normalized weight matrix:\n" + str(weightMatrix) + "\n")

	# Like Matrix is the likeliness between groups
	# LikeMat[2][3] is how much group 2 likes group 3
	# First row and column is the empty space
	# likeMatrix=np.array([[0.0, 0.0, 0.0],
	# 					[0.0, 0.7, 0.3],
	# 					[0.0, 0.3, 0.7]])

	# likeMatrix=[[1,0,0],
	# 			[0,1,0],
	# 			[0,0,1]]

	likeMatrix = np.array( [[0.0, 0.0, 0.0, 0.0, 0.0],
							[0.2, 1.0, 0.3, 0.3, 0.3],
							[0.1, 0.3, 1.0, 0.3, 0.3],
							[0.0, 0.3, 0.3, 1.0, 0.3],
							[0.0, 0.3, 0.3, 0.3, 1.0]])

	width = 150
	occupancyR = 0.8
	# groupRatios = [0.5,0.5]
	groupRatios = [0.6, 0.2, 0.15, 0.05]

	hThreshold = 0.5
	# hThreshold = np.percentile(happinessList, 20)
	moveSize = 100
	maxSteps = 200

	aggregationMap = InitializeMatrix(width, occupancyR, groupRatios)

	# get the percentiles and set the threshold accordingly
	happinessMatrix, unHappyL, vacancyL = CalcHappiness(aggregationMap, weightMatrix, likeMatrix, hThreshold)
	happinessList = HappinessListFromMatrix(happinessMatrix)
	print("Minimum happiness: {}".format(np.amin(happinessList)))

	percentileList = np.linspace(0, 100, num=11)
	for p in percentileList:
		val = np.percentile(happinessList, p)
		print("{}% percentile: {}".format(p, val))


	#############################################################################################
	# Plot
	#############################################################################################
	fig = plt.figure(figsize=(12,12))
	axMap = fig.add_subplot(2,2,1)
	axHappinessMap = fig.add_subplot(2,2,2)
	axHis = fig.add_subplot(2,2,3)
	axCur = fig.add_subplot(2,2,4)

	cmap = colors.ListedColormap(['white', 'green', 'red', 'blue', 'yellow'])
	ims = []
	stepCounter = -2

	def update(i):
		global aggregationMap, stepCounter

		happinessMatrix, unHappyL, vacancyL = CalcHappiness(aggregationMap, weightMatrix, likeMatrix, hThreshold)
		# avgHappiness = happinessMatrix.mean() / occupancyR
		hList = HappinessListFromMatrix(happinessMatrix)
		avgHappiness = hList.mean()

		if len(unHappyL) == 0:
			if stepCounter == -2:
				stepCounter = i
			print("Everyone's happy at step {}!".format(stepCounter))
			return fig
		aggregationMap = move(aggregationMap, unHappyL, vacancyL, moveSize)
		print("step {}".format(i))
		for ax in (axMap, axHis, axCur):
			ax.clear()
		axMap.imshow(aggregationMap, interpolation='nearest', origin='lower', cmap=cmap, animated=True)
		axMap.text(0.5, 1.05, "Step {}".format(i),
					size=plt.rcParams["axes.titlesize"],
					ha="center", transform=axMap.transAxes, )
		axHappinessMap.imshow(happinessMatrix, interpolation='nearest', origin='lower', animated=True)
		# axHis.hist(x4[:curr], normed=True, bins=np.linspace(14,20,num=21), alpha=0.5)
		weights = np.ones_like(hList) / float(len(hList))
		axHis.hist(hList, bins=np.linspace(0,1,21), weights=weights, alpha=0.8)
		axHis.text(0.5, -0.15, "Avg Happiness {}".format(round(avgHappiness, 3)),
					size=plt.rcParams["axes.titlesize"],
					ha="center", transform=axHis.transAxes, )
		return fig

	ani = animation.FuncAnimation(fig, update, frames=list(range(maxSteps)), interval=100, repeat=False)
	# plt.show()
	ani.save('./Videos/aggregation_w{}_occRate{}_th{}_groups{}.mp4'.format(width, occupancyR, round(hThreshold, 2), len(groupRatios)))
