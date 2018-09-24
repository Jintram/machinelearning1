# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:49:52 2018

@author: Jintram
"""

# Note that dataset location is:
# D:\Work\Udemy\Machine_Learning_Datasets\Part 1 - Data Preprocessing\Data.csv

datapath = "D:/Work/Udemy/Machine_Learning_Datasets/Part 1 - Data Preprocessing/"
datafile = "Data.csv"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# set work directory
os.chdir(datapath)

# 
dataset = pd.read_csv(datafile)

# now create matrix with our info
Xprime = dataset.iloc[:, :-1].values
    # take all lines, all columns except last one
X = Xprime
        
yprime = dataset.iloc[:, -1].values
y = yprime

################################################# Lecture 13

# Taking care of missing data
# one strategy is to replace missing data by mean of rest
# Use library for this
from sklearn.preprocessing import Imputer

# TODO: not sure about the details of this Imputer class

# make an imputer object
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
# configure it with (i.e. fit it to) this data
imputer = imputer.fit(X[:, 1:3 ])
# apply it to this data
X[:,1:3] = imputer.transform(X[:, 1:3 ])

################################################# Lecture 14

# What to do about categorical data? -> replace it by values
# Note that for category with multiple values, we need to 
# introduce dummy variables, ie give each category its own binary
# array

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

################################################# Lecture 15
# Splitting into training and test set

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 0)
                    # generally, 20-30% used for the test set
                    # Note that random_state is seed to get same Hadelyn    
                
                    
################################################# Lecture 16
# Feature scaling
                    
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
            # Note that fit_transform is two fns in one
X_test = sc_X.transform(X_test)
            # Here, also the dummy vars have been scaled, this is 
            # not strictly necessary, as they already are in a 
            # similar range.
            # It might be convenient to NOT scale them if you want
            # to be able to interpret your model better.

# Here, the dependent var y doesn't need scaling becaause it has two 
            # values only, but sometimes it is necesary.
























