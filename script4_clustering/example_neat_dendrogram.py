# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:00:21 2018

@author: Jintram
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# some example data to cluster
mypoints = np.array([[2.2,2], 
              [2.3,2.1], 
              [2.1,3], 
              [2.0,3.1], 
              [5.2,6],
              [5.3,6.1],
              [5.1,7],
              [5.0,7.1]])

# plot it
plt.plot(mypoints[:,0],mypoints[:,1],'o')
plt.xlim([0,10])
plt.ylim([0,10])
plt.show()


# dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(mypoints, method = 'ward'))
plt.xlabel('Customers')
plt.ylabel('Distances')
plt.show()