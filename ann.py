# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as p
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 3: 16].values
y = dataset.iloc[:, 16].values

# Encoding catgorical data - No categorical data present to encode

# Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling: To avoid one independent variable dominate another one
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# The dataset is now preprocessed

# Part 2 - Now let's make the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # Required to initialize the Neural Network
from keras.layers import Dense # Required for setting up the layers of ANN

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13)) # This method adds different layers to the neural network

# Adding the second hiden layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu')) # We added the input dim in the previous layer becoz the network had no idea of the number of input parameters to be expected

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test Set results
y_pred = classifier.predict(X_test)

# Here we dont have binary values so we set a threshold of 0.5
y_pred = (y_pred > 0.5)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Computing the accuracy
accuracy = ((cm[0][0] + cm[1][1]) / 2000) * 100