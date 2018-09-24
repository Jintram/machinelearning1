# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 12:51:23 2018

@author: Jintram
"""

# GENERAL ARCHITECTURE CNN
# ===
# 1. Convolution
# 2. Max pooling
# 3. Flattening
# 4. Full connection


# setting up the path
filepath1 = "C:/Users/Jintram/Temporary/cats_and_dogs_training/"
catfiles = "cats/"
dogfiles  = 'dogs/'

# 
from keras.models import Sequential
    # two types NNs: sequence of layers vs. graph, here we use sequence of ~
from keras.layers import Convolution2D
    # convolution for 2d
from keras.layers import MaxPooling2D 
    # pooling for 2d
from keras.layers import Flatten 
    # combine feature maps into 1 vector
from keras.layers import Dense 
    # conventional fully connected neural network that takes feature vector as input

# ===================================
# Setting up the CNN
# ===================================

# Initialize the CNN
# create object for linear stack of layers
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(8, (4,4) , input_shape = (64,64,3), activation='relu'))
    # *
    # Convolution2D(32, (3,3), ..) creates 32 feature maps of 3x3
    # *
    # NOTE about the padding option
    # ===
    # Same vs. valid concerns how the filter is shifted along the image and 
    # how it is handled if the image width is not a multiple of the filter's 
    # width.
    # Note that the option "same" takes into account all data in the image 
    # but uses padding, whilst valid doesn't use padding, but drops part
    # of the image.

# Now do the pooling (down-sizing the feature maps to reduce computational load)
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add a second convolutional layer
classifier.add(Convolution2D(4, (4,4) , activation='relu')) 
    # input shape is now known to Keras
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Flattening
classifier.add(Flatten())

# Conventional neural network at the end to process the features
# ===

# hidden layers
for i in range(3):
    classifier.add(Dense(output_dim = 8, activation = 'relu')) 
# output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) 

# Compiling the CNN
# ===

classifier.compile(optimizer='adam', loss='binary_crossentropy')#, metric=['accuracy'])
    # for >2 outcomes, use categorical_cross_entropy

# ===================================
# Training the CNN    
# ===================================
# Coding taken from
# https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
    
# Generate the training dataset object    
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# Generate the test dataset object
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the training dataset
training_set = train_datagen.flow_from_directory(
        filepath1 + "training_set/",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Create the test dataset
test_set = test_datagen.flow_from_directory(
        filepath1 + "test_set/",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# %% Actually fit the model (takes ages)

# Fit the model
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

# %% Also enable model plotting
# From tutorial https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
# Apparently, the graphviz package is required for this, and that needs 
# specific bindings, which are supposedly (only) configured properly when it is 
# installed the following way:
# conda install python-graphviz
# https://github.com/ContinuumIO/anaconda-issues/issues/1666
# However, that failed to work...

"""
from keras.utils.vis_utils import plot_model

# 
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
"""




























