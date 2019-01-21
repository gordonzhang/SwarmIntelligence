import numpy as np
from numpy import sin, cos, deg2rad
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = (2, 3)

rMat = np.array([[cos(deg2rad(-30)), -sin(deg2rad(-30))],
              [sin(deg2rad(-30)), cos(deg2rad(-30))]])
print(np.matmul(rMat, x))