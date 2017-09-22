#seed set
import numpy as np
np.random.seed(42)

#LoadDependency
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout # new!
from keras.layers.normalization import BatchNormalization # new!
from keras import regularizers # new! 
from keras.optimizers import SGD

#Load_data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Prepocess data
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
