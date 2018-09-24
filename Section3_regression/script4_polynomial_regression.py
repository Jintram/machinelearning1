# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:49:52 2018

@author: Jintram
"""

# Setup location of our datafile
datapath_root = "D:/Work/Udemy/Machine_Learning_Datasets/"
datapath_section = "Part 2 - Regression/Section 6 - Polynomial Regression/"
datafile = "Position_Salaries.csv"

# libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir(datapath_root+datapath_section) # set work directory

# Read data
dataset = pd.read_csv(datafile)

# now create matrix with our data
# ===
Xprime = dataset.iloc[:, 1:2].values
    # by saying 1:2 an array of arrays is produced
    # Could also use reshape
X = Xprime
        
yprime = dataset.iloc[:, -1].values
y = yprime

# Taking care of missing data
# ===
"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:, 1:3 ])
X[:,1:3] = imputer.transform(X[:, 1:3 ])

# Processing categorical data
# ===

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
"""

# Splitting into training and test set
# ===
"""
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 0)
                    # generally, 20-30% used for the test set
                    # Note that random_state is seed to get same Hadelyn    
"""
                
# Feature scaling
# ===

"""                    
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # Note that fit_transform is two fns in one
X_test = sc_X.transform(X_test) # Sometimes better to exclude dummy from scaling
"""

# Fitting
# ===
# Polynomial fitting by manually generating array
# [f1(x), f2(x), f3(x), .., fn(x)]
# Then using linear fit
# Preprocessing can be done automatically by PolynomialFeatures


# Import libs
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Only simple 1st order linear regression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Now with polynomial 
# Make polynomial 
DEGREE = 4;
poly_reg = PolynomialFeatures(degree=DEGREE)
X_poly = poly_reg.fit_transform(X)
# Regression
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


# Either use formula or again use polynomial transform to get single value
MYVALUE = 6.5
MYVALUE_poly = poly_reg.fit_transform([[MYVALUE]])
lin_reg2.predict(MYVALUE_poly)

myxrange = np.arange(0,10,0.1)
myxrange = myxrange.reshape(len(myxrange), 1)
myxrange_poly = poly_reg.fit_transform(myxrange)

# Plot
plt.plot(X,y,'-ob')
plt.plot(X,lin_reg.predict(X),'-or')
# plt.plot(X,lin_reg2.predict(X_poly),'-xr')
plt.plot(MYVALUE, lin_reg2.predict(MYVALUE_poly), 'sr',markersize=30,markerfacecolor="none")
plt.plot(MYVALUE, lin_reg2.predict(MYVALUE_poly), 'xr',markersize=10,markerfacecolor="none")
plt.text(MYVALUE+1, lin_reg2.predict(MYVALUE_poly),
         '%.0f' % lin_reg2.predict(MYVALUE_poly), 
         backgroundcolor = [1, 1, 1])
plt.plot(myxrange, lin_reg2.predict(myxrange_poly), '-k',markersize=10,markerfacecolor="none")

plt.show()












