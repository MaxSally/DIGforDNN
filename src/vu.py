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
    def ninps(self):
        assert self.model.input_shape[0] is None

        return self.model.input_shape[1]

    @property
    def symbolic_inps(self):
        return [z3.Real(f"i{n}") for n in range(self.ninps)]

    @property
    def symbolic_solver(self):
        try:
            return self._symbolic_solver
        except AttributeError:
            self._symbolic_solver = z3.Solver()
            self._symbolic_solver.add(self.symbolic_states)
            return self._symbolic_solver

    @property
    def symbolic_states(self):
        try:
            return self._symbolic_states
        except AttributeError:
            ss = []
            prev_nodes = self.symbolic_inps
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

    def seval(self, inps):
        """
        Apply constraint solving on symbolic states to obtain concrete values
        """

        assert isinstance(inps, list), inps
        assert len(inps) == self.ninps

        inps = z3.And([s == v for s, v in zip(self.symbolic_inps, inps)])

        self.symbolic_solver.push()
        self.symbolic_solver.add(inps)
        if self.symbolic_solver.check() == z3.sat:
            m = self.symbolic_solver.model()
        else:
            m = None
        print(self.symbolic_solver)
        self.symbolic_solver.pop()

        return m

    def ceval(self, inps):

        if isinstance(inps, list):
            inps = np.array([inps])
        assert len(inps[0]) == self.ninps

        # self.model.compile()
        extractor = keras.Model(inputs=self.model.inputs,
                                outputs=[layer.output for layer in self.model.layers])
        outputs = extractor(inps)
        print('outputs', outputs)
        return outputs

    def collect_traces(self, nsamples=5):
        # collect traces
        X = [[] for _ in range(len(self.model.layers)+1)]
        for test in range(nsamples):
            # todo: replace 1,2 with layer size
            inps = np.random.uniform(-10, 10, (1, self.ninps))
            X[0].append(inps[0])

            lid = 1
            outputs = self.ceval(inps)
            for layer in outputs:
                X[lid].append(layer.numpy()[0])
                lid += 1

        return X

    def infer(self, nsamples=5):
        X = self.collect_traces(nsamples)


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
# print(model.symbolic_states)
model = model_pa4()
# print(model.symbolic_states)
inp = [4.0, 3.5]
print(model.seval(inp))
print(model.ceval(inp))
# [n0_1 = 15/2,
#  o1 = 1/2,
#  o0 = -1/2,
#  i1 = 7/2,
#  n1_1 = 1/2,
#  n1_0 = 0,
#  i0 = 4,
#  n0_0 = 1/2]
inp = [1.0, -1.0]
print(model.seval(inp))
print(model.ceval(inp))
model.infer()

# print(model.symbolic_states)
# createAndSaveModelAsOnnx(json_file)
