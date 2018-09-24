# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 13:04:18 2018

@author: Jintram
"""

# Use 
# %matplotlib auto 
# and 
# %matplotlib inline
# To switch back and forth how plots are shown
# ----> auto makes it crash unfortunately


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

fig = plt.figure()

#ax = fig.add_subplot(111, projection='3d')

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
sigma=.7
Z = np.exp(-np.power(
        np.sqrt(np.power(X-2,2)+
        np.power(Y-2,2))
        ,2)/np.power((2*sigma),2))

ax = fig.gca(projection='3d')

ax.plot_surface(X, Y, Z)
plt.show()

print("Done")