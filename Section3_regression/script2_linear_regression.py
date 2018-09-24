# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 17:41:24 2018

@author: Jintram
"""

datapath = "D:/Work/Udemy/Machine_Learning_Datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/"
datafile = "Salary_Data.csv"


# libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir(datapath) # set work directory

# Read data
dataset = pd.read_csv(datafile)

# now create matrix with our data
Xprime = dataset.iloc[:, :-1].values
X = Xprime
        
yprime = dataset.iloc[:, -1].values
y = yprime

# Splitting into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 1/3, random_state = 0)
                    # generally, 20-30% used for the test set
                    # Note that random_state is seed to get same Hadelyn    
                
# Now fit the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Create the line
xToFit = np.array([[0], [max(X_train)]]) 
yFitted = regressor.predict(xToFit)
# Create predictions for test-set
y_pred = regressor.predict(X_train)

# Show the data
#plt.cla()
plt.scatter(X_train, y_train,color="red")
plt.plot(X_train, y_pred,"-b")
plt.title('Training set')
plt.xlabel('Experience (yrs)')
plt.ylabel('Salary ($)')
plt.show()


plt.scatter(X_test, y_test,color="red")
plt.plot(X_train, y_pred,"-b")
plt.title('Training set')
plt.xlabel('Experience (yrs)')
plt.ylabel('Salary ($)')
plt.show()
















