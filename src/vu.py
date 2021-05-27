# {"number_of_layer": 4, "number_of_neurons_each_layer": [2, 2, 2, 2], "weight": [[[1.0, 1.0], [-1.0, 1.0]], [[0.5, -0.5], [-0.2, 0.1]], [[1.0, -1.0], [-1.0, 1.0]]], "bias": [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]]}

import pdb
from Helper import createAndSaveModelAsOnnx
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

DBG = pdb.set_trace


def create_model(json_file):
    dtype = 'float64'

    with open(json_file) as f:
        data = json.load(f)
        weights = data["weight"]
        bias = data["bias"]

    print(bias)
    print(weights)
    model = Sequential()

    nlayers = len(weights)

    for layer in range(nlayers):
        weightsl = weights[layer]
        units = len(weightsl)
        kernel_initializer = tf.constant_initializer(weightsl)
        bias_initializer = tf.constant_initializer(bias[layer])
        print(weightsl, kernel_initializer)
        print(bias[layer], bias_initializer)
        # DBG()
        if layer == 0:
            d = Dense(units=units,
                      input_shape=(len(weightsl),),
                      activation='relu',
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      dtype=dtype
                      )
        elif layer == nlayers - 1:
            d = Dense(units=units,
                      activation=None,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      )
        else:
            d = Dense(units=units,
                      activation='relu',
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      )

        model.add(d)

    model.summary()
    inps = np.array([2, -2]).reshape(1, 2)
    print(inps)
    model.compile()
    print('output', model.evaluate(inps))

    print('layers', len(model.layers), model.layers)
    print('weights', len(model.weights), model.weights)
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    outputs = extractor(inps)
    print('outputs', outputs)

    model = Sequential()
    model.add(Dense(units=2,
                    input_shape=(2, ),
                    activation='relu',
                    kernel_initializer=tf.constant_initializer(
                        [[1.0, 1.0], [-1.0, 1.0]]),
                    bias_initializer=tf.constant_initializer([[0.0], [0.0]]),
                    dtype=dtype
                    ))
    model.add(Dense(units,
                    activation='relu',
                    kernel_initializer=tf.constant_initializer(
                        [[0.5, -0.5], [-0.2, 0.1]]),
                    bias_initializer=tf.constant_initializer([[0.0], [0.0]]),
                    dtype=dtype
                    ))
    model.add(Dense(units,
                    activation=None,
                    kernel_initializer=tf.constant_initializer(
                        [[1.0, -1.0], [-1.0, 1.0]]),
                    bias_initializer=tf.constant_initializer([[0.0], [0.0]]),
                    dtype=dtype
                    ))
    model.summary()
    inps = np.array([2, -2]).reshape(1, 2)
    print(inps)
    model.compile()
    print('output', model.evaluate(inps))

    return model


json_file = '../sample_input/json/sample_input_1.json'
create_model(json_file)

# createAndSaveModelAsOnnx(json_file)
