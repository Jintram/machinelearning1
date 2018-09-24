# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 17:41:24 2018

@author: Jintram
"""

datapath = "D:/Work/Udemy/Machine_Learning_Datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/"
datafile = "50_Startups.csv"


# libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir(datapath) # set work directory

# Read data
dataset = pd.read_csv(datafile)

# now create matrix with our data
Xprime = dataset.iloc[:, 0:4].values
X = Xprime
        
yprime = dataset.iloc[:, -1].values
y = yprime

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean', axis=0)
imputer = imputer.fit(X[:, 0:3 ])
X[:,0:3] = imputer.transform(X[:, 0:3 ])


# Processing categorical data
# ===

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Transform to numbers  
X[:,3] = labelencoder_X.fit_transform(X[:,3])
# Now create dummy vars
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Remember that "dummy variable trap" exists, i.e. one of the categories' 
# binary values should be removed. This is however done by the libraries
# that will be used.
X = X[:,1:]

"""
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)    
"""


# Splitting into training and test set
# ===
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = .2, random_state = 0)
                    # generally, 20-30% used for the test set
                    # Note that random_state is seed to get same Hadelyn    

            
                
# Now fit the data, using ALL dependent parameters
# ===
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Create predictions for test-set
y_pred = regressor.predict(X_test)

width=.35
#fig, ax = plt.subplots()
plt.bar(np.arange(len(y_test)),y_test,width)
plt.bar(np.arange(len(y_test))+width,y_pred,width)
plt.show()


# Now apply some predictor selection
# Using backward elimation
# ===

import statsmodels.formula.api as sm

# add offset param
X_train_opt =  np.append(np.ones([len(X_train),1]), X_train, axis = 1 )
X_test_opt =  np.append(np.ones([len(X_test),1]), X_test, axis = 1 )

# First fit
regressor_OLS = sm.OLS(endog=y_train, exog=X_train_opt).fit()
regressor_OLS.summary()


goflag = 1; toremove=[]
ptreshold=0.5
while goflag:
    
    
    # get pmax
    mypvalues = regressor_OLS.pvalues
    idxMaxP = np.argmax(mypvalues)
    
    print("Identified #" + str(idxMaxP) + " as max idx.")
    
    # if above treshold remove
    if mypvalues[idxMaxP]>ptreshold and np.size(X_train_opt,1)>2:
    
        print("Above treshold so removing.")
        
        # Remove the unnecessary column
        X_train_opt = np.delete(X_train_opt, toremove, 1)
        X_test_opt = np.delete(X_test_opt, toremove, 1)
        
        # And fit again
        regressor_OLS = sm.OLS(endog=y_train, exog=X_train_opt).fit()
        
        # Note down which ones to remove
        toremove.append(idxMaxP)
        
    else:
        
        print("Not above treshold and/or end of sequence reached")
        
        goflag = 0

print(str(len(toremove)) + " features were removed.")

# (Note that it turns out a single parameter is the only significant one.)
    


# Now plot the outcome again
# ===
        
# Create predictions for test-set
y_pred_OLS = regressor_OLS.predict(X_test_opt)

width=.25
fig, ax = plt.subplots()
ax.bar(np.arange(len(y_test)),y_test,width)
ax.bar(np.arange(len(y_test))+width,y_pred,width)
ax.bar(np.arange(len(y_test))+2*width,y_pred_OLS,width)
plt.show()    
      
# Note that one by one there's always a parameter that is insignificant
# Perhaps not all assumptions were met?













