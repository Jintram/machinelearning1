# Apriori

# Setup location of our datafile
datapath_root = "D:/Work/Udemy/Machine_Learning_Datasets/"
datapath_section = "Part 5 - Association Rule Learning/Section 28 - Apriori/"
datafile = "Market_Basket_Optimisation.csv"

import os
os.chdir(datapath_root+datapath_section)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(datafile, header = None)

transactions = []
for i in range(0, 7501):
     #transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
     transactions.append(   dataset.values[i,~pd.isnull(dataset.values[i])].tolist()   )
    

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)