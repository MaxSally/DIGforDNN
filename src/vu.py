import pdb
from Helper import createAndSaveModelAsOnnx
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

    def get_symbolic_states(self):
        #mylayer = self.model.layers[0]
        # print(mylayer)
        # print(mylayer.weights)
        # self.model.summary()
        #keras.utils.plot_model(self.model, "model.png")

        # print(self.model.input_shape)
        # print(self.model.output_shape)

        prev_layer = [z3.Real(f"i{n}")
                      for n in range(self.model.input_shape[1])]
        ss = []
        for lid, layer in enumerate(self.model.layers):
            # node_str = 'o' if m == len(self.model.layers) - 1 else f'n{m}'
            cur_layer = [z3.Real(f"n{lid}_{nid}" if lid < len(self.model.layers) - 1 else f"o{nid}")
                         for nid in range(len(range(layer.output_shape[1])))]

            layer_config = layer.get_config()

            prev_layer_ = np.array([prev_layer])

            #print('prev', prev_layer_.shape, prev_layer_)

            #print('weight,bias\n', layer.get_weights())

            weights, biases = layer.get_weights()
            #print('weight', weights.shape, weights)
            #print('bias', biases.shape, biases)
            #print('weights', weights.shape, weights)
            v = prev_layer_.dot(weights) + biases
            #print(v, biases)
            v = v[0]

            # print(v)
            #print('cur_layer', cur_layer)
            activation = layer_config['activation']
            assert len(cur_layer) == len(v)
            for i in range(len(cur_layer)):
                v_ = v[i]
                if activation == 'relu':
                    v_ = z3.If(0 >= v_, 0, v_)
                v_ = z3.simplify(v_)
                ss.append(cur_layer[i] == v_)
            prev_layer = cur_layer

        f = z3.simplify(z3.And(ss))
        print(f)
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
                    activation='relu',
                    kernel_initializer=tf.constant_initializer(
                        [[1.0, 1.0], [-1.0, 1.0]]),
                    bias_initializer=tf.constant_initializer([[0.0], [0.0]]),
                    dtype=dtype
                    ))
    model.add(Dense(units=2,
                    activation='relu',
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

    #n00, n01
    d0 = Dense(units=2,
               input_shape=(3, ),
               activation='relu',
               kernel_initializer=tf.constant_initializer(
                   [[1.0, 1.0], [-1.0, 1.0],  [1.0, -1.0]]),
               bias_initializer=tf.constant_initializer(
                   [[0.0], [0.0]]),
               dtype=dtype)
    model.add(d0)

    # n10,n11,n12
    d1 = Dense(units=3,
               activation='relu',
               kernel_initializer=tf.constant_initializer(
                   [[0.5, -0.5, 0.3], [-0.2, 0.1, -0.3]]),
               bias_initializer=tf.constant_initializer([[0.0], [0.0], [0.0]]),
               dtype=dtype
               )
    model.add(d1)

    # n20, n21
    d2 = Dense(units=2,
               activation='relu',
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
    # print(model.layers)
    # print(model.weights)
    return Model(model)


# def create_model(json_file):

#     with open(json_file) as f:
#         data = json.load(f)
#         weights = data["weight"]
#         bias = data["bias"]

#     print(bias)
#     print(weights)
#     model = Sequential()

#     nlayers = len(weights)

#     for layer in range(nlayers):
#         weightsl = weights[layer]
#         units = len(weightsl)
#         kernel_initializer = tf.constant_initializer(weightsl)
#         bias_initializer = tf.constant_initializer(bias[layer])
#         print(weightsl, kernel_initializer)
#         print(bias[layer], bias_initializer)
#         # DBG()
#         if layer == 0:
#             d = Dense(units=units,
#                       input_shape=(len(weightsl),),
#                       activation='relu',
#                       kernel_initializer=kernel_initializer,
#                       bias_initializer=bias_initializer,
#                       dtype=dtype
#                       )
#         elif layer == nlayers - 1:
#             d = Dense(units=units,
#                       activation=None,
#                       kernel_initializer=kernel_initializer,
#                       bias_initializer=bias_initializer,
#                       )
#         else:
#             d = Dense(units=units,
#                       activation='relu',
#                       kernel_initializer=kernel_initializer,
#                       bias_initializer=bias_initializer,
#                       )
#         model.add(d)

#     return model


# json_file = '../sample_input/json/sample_input_1.json'
# create_model(json_file)

# model = model_pa4()
# inps = np.array([4, 3.5]).reshape(1, 2)
# model.meval(inps)
# model.infer()


model = model_pa5()
model.get_symbolic_states()
model = model_pa4()
model.get_symbolic_states()
# createAndSaveModelAsOnnx(json_file)
