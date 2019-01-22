import numpy as np
from numpy import sin, cos, deg2rad
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def HappinessListOfEachGroup(hMat: np.ndarray, aggMap: np.ndarray, gRatios: list):
	nG = len(gRatios)
	hList = []
	for i in range(nG):
		iG = i + 1
		index = np.where(aggMap == iG)
		hList.append(hMat[index].reshape(-1))
	return hList


x = np.array([[1,2,3],[11,1,13],[21,22,23]])
y = np.array([[99,81,77],[11,12,13],[21,22,23]])

yy = HappinessListOfEachGroup(y, x, [1,1])
print(yy)
