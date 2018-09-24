# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# setting up the path
filepath1 = "C:/Users/Jintram/Documents/Udemy/ML_datasets_at_laptop/"
filepath2 = "Artificial_Neural_Networks/"
filename  = 'Churn_Modelling.csv'

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# %% Importing the dataset
dataset = pd.read_csv(filepath1+filepath2+filename)
X = dataset.iloc[:, 3:13].values
X_orig = X # keep original data also
y = dataset.iloc[:, 13].values

# %% Encoding categorical data
# categories -> numbers, numbers -> dummies, retain info these are categorical,
# remove one of the dummy fields because of redundancy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) 
    # Create dummy params here, note not needed for other since values are {0,1}
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # removes redundant data column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (very important in neural networks)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# %% Let's take a look at the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_orig[y==0,0],X_orig[y==0,3],zs=X_orig[y==0,4],c='b', alpha=.1)
ax.scatter(X_orig[y==1,0],X_orig[y==1,3],zs=X_orig[y==1,4],c='r', alpha=.1)
plt.show()

# %% Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # initializes neural network
from keras.layers import Dense # build layers

# Initialising the ANN
classifier = Sequential()

# rule of thumb: nr of nodes in hidden layers is mean of nr of input and output
# parmaeters

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # adam is a gradient descent algorithm
    # loss function is for gradient descent, to optimize; can use sth simple but also log fn 
    # note binary simply indicates output type
    # accuracy is standard choice (is simply nr_pred_correct/nr_all)
    # ---
    # I think the metrics determines how the composite y value is determined,
    # and the loss function how the deviation from that y value is quantified

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
    # batch_size is nr of observations after which weights are updated
    # epoch is full round

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)