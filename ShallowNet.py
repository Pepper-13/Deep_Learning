#Set seed for reproducibility 
import numpy as np
np.random.seed(42)

#Load dependencies
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Have a look at your data and its shape
X_train.shape
#o/p = (6000,28,28)
y_train.shape
#o/p = (6000,)
y_train[0:99]
x_train[0]
y_test.shape
X_test.shape
#o/p = (10000,28,28)

#Preprocess the data
#Change the two dimensional data to one dimensional data
X_train = X_train.reshape(60000,784).astype('float32')
X_test = X_test.reshape(10000,784).astype('float32')
