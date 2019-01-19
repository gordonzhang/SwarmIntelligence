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


def CalcHappiness(aggMap:np.array, weightMat:np.array, likeMat:np.array, threshold:float):
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
				happinessScale = happVal / maxHappiness
				happinessMat[i,j] = happinessScale
				if happinessScale < threshold:
					unHappyList.append([i,j])

	return happinessMat, vacancyList, unHappyList


def move(aggMap:np.array, unHappyList:list, vacancyList:list, movingSize:int):
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


if "__main__" == __name__:
	# Weights of impact from neighbors, this person is in the center of te matrix.
	# Rows and columns must be odd numbers.

	# Example in class
	# weightMat = np.array([[1,1,1],
	# 					  [1,0,1],
	# 					  [1,1,1]])

	# Manhattan distance of 2, impact decreases by distance
	weightMatrix =np.array([[0.0,0.0,0.5,0.0,0.0],
							[0.0,0.5,1.0,0.5,0.0],
							[0.5,1.0,0.0,1.0,0.5],
							[0.0,0.5,1.0,0.5,0.0],
							[0.0,0.0,0.5,0.0,0.0]])

	# Like Matrix is the likeliness between groups
	# LikeMat[2][3] is how much group 2 likes group 3
	# First row and column is the empty space
	likeMatrix=[[0.0, 0.0, 0.0],
				[0.0, 0.7, 0.3],
				[0.0, 0.3, 0.7]]

	# likeMatrix=[[1,0,0],
	# 			[0,1,0],
	# 			[0,0,1]]

	width = 100
	occupancyR = 0.9
	groupRatios = [.5,.5]
	hThreshold = 0.5
	moveSize = 100
	maxSteps = 30

	aggregationMap = InitializeMatrix(width, occupancyR, groupRatios)
	print(aggregationMap)

	# get the percentiles and set the threshold accordingly
	happinessMatrix, vacancyL, unHappyL = CalcHappiness(aggregationMap, weightMatrix, likeMatrix, hThreshold)
	happinessList = happinessMatrix.reshape(-1)
	index = np.argwhere(happinessList==0)
	happinessList = np.delete(happinessList, index)
	print("Minimum happiness: {}".format(np.amin(happinessList)))

	percentileList = np.linspace(0, 100, num=11)
	for p in percentileList:
		val = np.percentile(happinessList, p)
		print("{}% percentile: {}".format(p, val))

	hThreshold = np.percentile(happinessList, 20)

	#############################################################################################
	# Plot
	#############################################################################################
	fig = plt.figure(figsize=(10,5))
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)

	ims = []
	cmap = colors.ListedColormap(['white', 'cyan', 'blue', 'green'])

	for s in range(maxSteps):

		happinessMatrix, vacancyL, unHappyL = CalcHappiness(aggregationMap, weightMatrix, likeMatrix, hThreshold)
		avgHappiness = happinessMatrix.mean() / occupancyR
		if len(unHappyL) == 0:
			print("Everyone's happy! Step: {}".format(s))
			break
		aggregationMap = move(aggregationMap, unHappyL, vacancyL, moveSize)
		print("step: {}".format(s))
		# print(aggregationMap)
		title1 = ax1.text(0.5,1.05,"Step: {}, Avg Happiness: {}".format(s, round(avgHappiness,3)),
						size=plt.rcParams["axes.titlesize"],
						ha="center", transform=ax1.transAxes, )
		im1 = ax1.imshow(aggregationMap, interpolation='nearest', origin='lower', cmap=cmap, animated=True)
		# im2 = ax2.hist(np.linspace(0, 100, num=100))
		ims.append([im1, title1])

	ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat=True)
	# ani.save('aggregation_w{}_occRate{}_th{}.mp4'.format(width, occupancyR, round(hThreshold,2)))
	plt.show()

	# fig = plt.figure(figsize=(6,8))
	# axMap = fig.add_subplot(2,1,1)
	# axHis = fig.add_subplot(2,2,3)
	# axCur = fig.add_subplot(2,2,4)
	#
	# ims = []
	# cmap = colors.ListedColormap(['white', 'cyan', 'blue', 'green'])
	#
	# def updateData(i):
	# 	if i > maxSteps:
	# 		break
	# 	for ax in (ax1, ax2, ax3, ax4):
	# 		ax.clear()
	# 	ax1.hist(x1[:curr], normed=True, bins=np.linspace(-6,1, num=21), alpha=0.5)
	# 	ax2.hist(x2[:curr], normed=True, bins=np.linspace(0,15,num=21), alpha=0.5)
	# 	ax3.hist(x3[:curr], normed=True, bins=np.linspace(7,20,num=21), alpha=0.5)
	# 	ax4.hist(x4[:curr], normed=True, bins=np.linspace(14,20,num=21), alpha=0.5)
	#
	# ani = animation.FuncAnimation(fig, updateData, interval=100, repeat=False)
	#
	# plt.show()
