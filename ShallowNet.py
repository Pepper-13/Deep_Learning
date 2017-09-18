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
