# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 13:11:49 2018

@author: Jintram
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

fig = plt.figure()

#ax = fig.add_subplot(111, projection='3d')

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = 1/(1+np.exp(-X))+1/(1+np.exp(-Y))

ax = fig.gca(projection='3d')

ax.plot_surface(X, Y, Z)
plt.show()

print("Done")