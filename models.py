import tensorflow as tf
import keras as k
from keras.layers import *
from keras.models import Sequential

baseline = Sequential([
    LSTM(4),
    Dense(1, activation='sigmoid')
])
