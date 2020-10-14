import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from keras import backend as K

model = Sequential()
weight = np.array([[[1.0, -1.0], [1.0, -1.0]], [[[0.5, -0.5], [-0.2, 0.1]]], [[[1.0, -1.0], [-1.0, 1.0]]]])
bias = np.array([[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]])
activation = 'relu'
layer1 = Dense(2,  input_shape=(2, ), activation=activation, kernel_initializer=tf.constant_initializer(weight[0]),
               bias_initializer=tf.constant_initializer(bias[0]))
layer2 = Dense(2, activation=activation, kernel_initializer=tf.constant_initializer(weight[1]),
               bias_initializer=tf.constant_initializer(bias[1]))
layer3 = Dense(2, kernel_initializer=tf.constant_initializer(weight[2]),
               bias_initializer=tf.constant_initializer(bias[2]))
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.summary()

inps = np.array([[-9, 10]])
inps.reshape(1, 2)
# for layer in model.layers:
#     keras_function = K.function([model.input], [layer.output])
#     outputs.append(keras_function([inps, 1]))
extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])
outputs = extractor(inps)
for layer in outputs:
    print(layer.numpy())


