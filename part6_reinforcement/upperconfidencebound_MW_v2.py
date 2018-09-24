# Apriori

# Setup location of our datafile
datapath_root = "D:/Work/Udemy/Machine_Learning_Datasets/"
datapath_section = "Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/"
datafile = "Ads_CTR_Optimisation.csv"

import os
os.chdir(datapath_root+datapath_section)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(datafile) # , header = None

# Visualize some data
adMeans = [np.mean(dataset.iloc[:,i]) for i in range(np.size(dataset,1))]
plt.bar(range(len(adMeans)),adMeans)
plt.xlabel("Ad")
plt.ylabel("Average click yes/no")
plt.show()

# Upper confidence bound algorithm
N = 10000
d = 10
total_rewards = np.zeros(10)
total_chosen  = np.zeros(10)
averages = np.ones(10)
confidences = np.ones(10)*np.inf
for n in range(0, N): # N
    
    # select an ad
    ad = np.argmax(averages+confidences)
    
    # update the observations with the currently selected add
    total_rewards[ad] += dataset.iloc[n,ad]
    total_chosen[ad] += 1
    
    # update averages
    averages[ad] = total_rewards[ad]/total_chosen[ad]
    
    # update confidences
    confidences[ad] = np.sqrt(   3*np.log(n+1) / (2*total_chosen[ad])   )
    
# step 2, profit
grandtotal_reward = np.sum(total_rewards)
# Hadelin has 2178 as outcome..
    
# Visualising the results
plt.bar(range(len(total_chosen)),total_chosen)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()



