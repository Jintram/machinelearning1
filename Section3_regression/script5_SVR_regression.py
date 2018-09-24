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
"""

# Processing categorical data
# ===
"""
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
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X) 
y = sc_y.fit_transform(y.reshape(-1, 1)) 


# Fitting
# ===
# Create regressor
from sklearn.svm import SVR
regressor = SVR(kernel="rbf") # rbf kernel is gaussian method
regressor.fit(X,y)

# Predicting new result
# ===
MYVALUE=sc_X.transform(np.array([[6.5]]))
y_pred = regressor.predict(MYVALUE)


# Either use formula or again use polynomial transform to get single value

myxrange = np.arange(min(X),max(X),0.1)
myxrange = myxrange.reshape(len(myxrange), 1)


# Plot
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(y),'ob')
plt.plot(sc_X.inverse_transform(myxrange),sc_y.inverse_transform(regressor.predict(myxrange)),'--r')
plt.show()

# print("Predicted value for {:.1f} is {:.1f}".format(MYVALUE, y_pred) )

print("Predicted value for "+str(MYVALUE)+" is "+str(np.around(sc_y.inverse_transform(y_pred)))+".")


"""
plt.plot(X,regressor.predict(X),'-or')
plt.plot(MYVALUE, lin_reg2.predict(MYVALUE_poly), 'sr',markersize=30,markerfacecolor="none")
plt.plot(MYVALUE, lin_reg2.predict(MYVALUE_poly), 'xr',markersize=10,markerfacecolor="none")
plt.text(MYVALUE+1, lin_reg2.predict(MYVALUE_poly),
         '%.0f' % lin_reg2.predict(MYVALUE_poly), 
         backgroundcolor = [1, 1, 1])
plt.plot(myxrange, lin_reg2.predict(myxrange_poly), '-k',markersize=10,markerfacecolor="none")

"""








