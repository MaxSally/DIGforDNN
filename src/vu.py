import pdb
from Helper import createAndSaveModelAsOnnx
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import activations
import numpy as np
import z3

DBG = pdb.set_trace

dtype = 'float64'


class Model:
    def __init__(self, model):
        self.model = model

    @property
    def nlayers(self):
        return len(self.model.layers)

    @property
    def symbolic_states(self):
        try:
            return self._symbolic_states
        except AttributeError:
            prev_nodes = [z3.Real(f"i{n}")
                          for n in range(self.model.input_shape[1])]
            ss = []
            for lid, layer in enumerate(self.model.layers):
                weights, biases = layer.get_weights()
                vs = np.array([prev_nodes]).dot(weights) + biases
                vs = vs[0]

                nnodes = layer.output_shape[1]  # number of nodes in layer
                assert nnodes == len(vs)
                activation = layer.get_config()['activation']
                cur_nodes = []
                for i in range(nnodes):
                    v = vs[i]
                    if activation == 'relu':
                        v = z3.If(0 >= v, 0, v)
                    v = z3.simplify(v)

                    if lid < len(self.model.layers) - 1:
                        node = z3.Real(f"n{lid}_{i}")
                    else:
                        node = z3.Real(f"o{i}")
                    cur_nodes.append(node)
                    ss.append(node == v)

                prev_nodes = cur_nodes

            f = z3.simplify(z3.And(ss))
            self._symbolic_states = f
            return f

    def meval(self, inps):

        print(inps)
        self.model.compile()
        # print('output', self.model.evaluate(inps))

        # print('layers', len(self.model.layers), self.model.layers)
        # print('weights', len(self.model.weights), self.model.weights)
        # print("\n")

        extractor = keras.Model(inputs=self.model.inputs,
                                outputs=[layer.output for layer in self.model.layers])
        outputs = extractor(inps)
        print('outputs', outputs)
        return outputs

    def infer(self):
        ntests = 500

        for test in range(ntests):
            # todo: replace 1,2 with layer size
            inps = np.random.uniform(-10, 10, (1, 2))
            print('myinps', inps, inps[0])
            outputs = self.meval(inps)
            DBG()


def model_pa4():
    model = Sequential()
    model.add(Dense(units=2,
                    input_shape=(2, ),
                    activation=activations.relu,
                    kernel_initializer=tf.constant_initializer(
                        [[1.0, 1.0], [-1.0, 1.0]]),
                    bias_initializer=tf.constant_initializer([[0.0], [0.0]]),
                    dtype=dtype
                    ))
    model.add(Dense(units=2,
                    activation=activations.relu,
                    kernel_initializer=tf.constant_initializer(
                        [[0.5, -0.5], [-0.2, 0.1]]),
                    bias_initializer=tf.constant_initializer([[0.0], [0.0]]),
                    dtype=dtype
                    ))
    model.add(Dense(units=2,
                    activation=None,
                    kernel_initializer=tf.constant_initializer(
                        [[1.0, -1.0], [-1.0, 1.0]]),
                    bias_initializer=tf.constant_initializer([[0.0], [0.0]]),
                    dtype=dtype
                    ))
    # model.summary()
    # print(model.layers)
    # print(model.weights)
    return Model(model)


def model_pa5():
    """
    # bias = # units
    # weights = # inputs (from prev layer)
    """

    model = Sequential()

    # n00, n01
    d0 = Dense(units=2,
               input_shape=(3, ),
               activation=activations.relu,
               kernel_initializer=tf.constant_initializer(
                   [[1.0, 1.0], [-1.0, 1.0],  [1.0, -1.0]]),
               bias_initializer=tf.constant_initializer(
                   [[0.0], [0.0]]),
               dtype=dtype)
    model.add(d0)

    # n10,n11,n12
    d1 = Dense(units=3,
               activation=activations.relu,
               kernel_initializer=tf.constant_initializer(
                   [[0.5, -0.5, 0.3], [-0.2, 0.1, -0.3]]),
               bias_initializer=tf.constant_initializer([[0.0], [0.0], [0.0]]),
               dtype=dtype
               )
    model.add(d1)

    # n20, n21
    d2 = Dense(units=2,
               activation=activations.relu,
               kernel_initializer=tf.constant_initializer(
                   [[0.1, -0.5], [0.2, 0.7], [1.2, -0.8]]),
               bias_initializer=tf.constant_initializer([[0.0], [0.0]]),
               dtype=dtype
               )
    model.add(d2)

    # o0, o1
    d3 = Dense(units=2,
               activation=None,
               kernel_initializer=tf.constant_initializer(
                   [[1.0, -1.0], [-1.0, 1.0]]),
               bias_initializer=tf.constant_initializer([[0.0], [0.0]]),
               dtype=dtype
               )
    model.add(d3)
    return Model(model)


model = model_pa5()
print(model.symbolic_states)
model = model_pa4()
print(model.symbolic_states)
print(model.symbolic_states)
# createAndSaveModelAsOnnx(json_file)
