import numpy as np
import json
import onnx
from onnx2keras import onnx_to_keras
import onnxmltools
import keras2onnx

from Model import createModel


def input_processing_json(file_path):
    '''
    Process input from json file.

    Parameters:
    -----------
    file_path: path to json input file.

    Return
    -----------
    number_of_layer: Number of layers in the neural network
    number_of_neurons_each_layer: list of number of neurons in each layer
    weight: list of weights of the neurons network
    bias: list of bias at each layer
    '''
    data = {}
    number_of_layer = -1
    number_of_neurons_each_layer = []
    weight = []
    bias = []
    with open(file_path, "r") as f:
        data = json.load(f)
        weight = data["weight"]
        bias = data["bias"]
        number_of_neurons_each_layer = data["number_of_neurons_each_layer"]
        number_of_layer = data["number_of_layer"]
    return number_of_layer, number_of_neurons_each_layer, weight, bias


def check_input_constraint(test_input, weight, bias, trace):
    '''
    :param test_input: genereated inputs
    :param weight: weight of the first layer
    :param bias: bias of the first layer
    :param trace: proposed properties
    :return:
    True or False whether the test_input satisfies the trace.
    '''
    second_layer = np.cross(test_input, weight) > 0
    return second_layer == trace


def input_processing_onnx(file_path=None):
    '''
    Process input from onnx file.

    Parameters:
    -----------
    file_path: path to onnx input file. (still in developed)

    Return
    -----------
    number_of_layer: Number of layers in the neural network
    number_of_neurons_each_layer: list of number of neurons in each layer
    weight: list of weights of the neurons network
    bias: list of bias at each layer
    ../sample_input/eranmnist_benchmark/onnx/tf/mnist_relu_3_50.onnx
    '''
    onnx_model = onnx.load(file_path)
    onnx_model.graph.node[2].output = 'dense/Relu_0'
    print(onnx_model.graph.node[2].output)
    # print(onnx_model.graph.input)
    exit(0)
    kera_model = onnx_to_keras(onnx_model, input_names=['dense_input'])
    number_of_layer = 0
    number_of_neurons_each_layer = []
    weight = []
    bias = []
    for layer in kera_model.layers:
        layer_weight = layer.get_weights()
        if not layer_weight:
            continue
        number_of_layer += 1
        if number_of_layer == 1:
            number_of_neurons_each_layer.append(
                np.array(layer_weight[0]).shape[0])
        number_of_neurons_each_layer.append(np.array(layer_weight[0]).shape[1])
        weight.append(np.array(layer_weight[0]))
        bias.append(np.array(layer_weight[1]))
    number_of_layer += 1
    print(number_of_layer)
    print(number_of_neurons_each_layer)
    print(weight)
    print(bias)
    return number_of_layer, number_of_neurons_each_layer, weight, bias


def saveModelAsOnnx(model, filename):
    onnx_model = onnxmltools.convert_keras(model, target_opset=7)
    onnxmltools.save_model(onnx_model, filename)


def createAndSaveModelAsOnnx(filename):
    number_of_layer, number_of_neurons_each_layer, weight, bias = input_processing_json(
        filename)
    model = createModel(
        number_of_layer, number_of_neurons_each_layer, weight, bias)
    #saveModelAsOnnx(model, filename.replace('.json', '.onnx').replace('json', 'onnx'))
