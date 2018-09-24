# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:26:03 2018

@author: Jintram
"""

# Just some testing
# Let's try a fibonacci

mySeries = [1,1]
desiredLength = 100;

for i in range(desiredLength-len(mySeries)):

    mySeries.append(mySeries[-2]+mySeries[-1])
    

print(mySeries)
