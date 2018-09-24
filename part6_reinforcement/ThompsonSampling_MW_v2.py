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
import random
N = 10000
d = 10
total_rewards = np.zeros(d)
total_chosen  = np.zeros(d)
for n in range(0, N): # N
    
    # Draw from our current distributions, which come from Bayesian inference
    random_betas = [random.betavariate(total_rewards[i]+1, total_chosen[i]-total_rewards[i]+1)
                    for i in range(d)]
    
    # select an ad
    ad = np.argmax(random_betas)
        
    # update the observations with the currently selected add
    total_rewards[ad] += dataset.iloc[n,ad]
    total_chosen[ad] += 1

    
# step 2, profit
grandtotal_reward = np.sum(total_rewards)
    
# Visualising the results
plt.bar(range(len(total_chosen)),total_chosen)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

print("We earned $" + str(grandtotal_reward) + ".")

