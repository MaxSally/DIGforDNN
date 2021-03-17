import os
import sys
from copy import deepcopy

import onnxruntime as onnxruntime
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import json
import onnx
from onnx_tf.backend import prepare
import onnx2keras
from onnx2keras import onnx_to_keras
import z3


class NodePath:
    name = ""
    threshold = 0
    sign = 0

    def __init__(self, name, threshold, sign):
        self.name = name
        self.threshold = threshold
        self.sign = sign


def printDecisionTree(tree, feature_names):
    '''
    Outputs decision tree in text representation

    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as pseudocode
    feature_names: list
        The feature names of the dataset used for building the decision tree
    '''
    text_representation = tree.export_text(tree, feature_names=feature_names)
    text_representation = text_representation.replace('<= 0.50', '== FALSE')
    text_representation = text_representation.replace('>  0.50', '== TRUE')
    print(text_representation)


def extract_decision_tree(tree, feature_names):
    '''
    Outputs a decision tree model as if/then pseudocode

    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as pseudocode
    feature_names: list
        The feature names of the dataset used for building the decision tree
    '''

    left_subtree = tree.tree_.children_left
    right_subtree = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    path = []
    result = []

    def recurse(left, right, threshold, features, node, path, depth=0):
        if threshold[node] != -2:
            if left[node] != -1:
                originalPath = deepcopy(path)
                path.append(NodePath(features[node], threshold[node], 0))
                recurse(left, right, threshold, features, left[node], path,
                        depth + 1)
                if right[node] != -1:
                    originalPath.append(NodePath(features[node], threshold[node], 1))
                    recurse(left, right, threshold, features, right[node],
                            originalPath, depth + 1)
        else:
            if len(value[node][0]) == 1:
                return
            if value[node][0][0] < value[node][0][1] and value[node][0][0] == 0:
                case = dict()
                for nodePath in path:
                    case[nodePath.name] = [round(nodePath.threshold, 2), nodePath.sign]
                result.append(case)

    recurse(left_subtree, right_subtree, threshold, features, 0, path)
    return result


def get_neuron_name_of_layer(layer):
    '''
    Outputs a list of neuron names of a layer

    Parameters:
    -----------
    layer: layer that user wants to retrieve names for
    '''
    names = []
    NEURON = ""
    if layer == 0:
        NEURON = "x"
    elif layer == number_of_layer - 1:
        NEURON = "y"
    else:
        NEURON = "NEURON_" + str(layer)
    for i in range(number_of_neurons_each_layer[layer]):
        names.append(NEURON + "_" + str(i))
    return names


def getY_implication_for_final_layer(rule, Y):
    '''
    Outputs a list of True if rule is satisfied, false otherwise for each case.
    Noted: each set of inputs including subsequent outputs at each neurons is considered a case.

    Parameters:
    -----------
    rule: rule to considered
        rule 0 means that y0 is the largest among all output neurons.
    Y: 2-D array. List of outputs.
        Each Y[i] contains a list which is the result of each neurons in case i.
    '''
    local_Y = []
    for example in Y:
        local_Y.append(example[rule] > max([x for i, x in enumerate(example) if i != rule]))
    return local_Y


def getY_each_layer_implication_activation(rule, Y):
    '''
    Outputs a list of True if rule is satisfied, false otherwise for each case.
    Noted: each set of inputs including subsequent outputs at each neurons is considered a case.

    Parameters:
    -----------
    rule: rule to considered
        rule 0 means that y0 is the largest among all output neurons.
    Y: 2-D array. List of outputs.
        Each Y[i] contains a list which is the result of each neurons in case i.
    '''
    local_Y = []
    for example in Y:
        local_Y.append(example[rule] > 0)
    return local_Y


def get_implication_in_text(layer, rule):
    '''
    Outputs implication in text (" => (NEURON_NAME)_(LAYER)_(ORDER_OF_NEURON_IN_THAT_LAYER)

    Parameters:
    -----------
    layer: layer to be considered for the rule
    rule: rule to considered
        rule 0 means that y0 is the largest among all output neurons.
    '''
    implication = ""
    NEURON_NAME = ("NEURON_" + str(layer) if layer < number_of_layer - 1 else "y")
    implication = NEURON_NAME + "_" + str(rule)
    return implication


def print_implication_between_two_layers(weight, bias, neuron, names, layerI, rule):
    '''
    Outputs implication between two layers in text (layerI and layerI + 1

    Parameters:
    -----------
    weight: weight between two layers
    bias: bias between two layers
    neuron: contains all neurons to be printed for a specific rule and their result from Decision Tree
    names: names of neuron in layerI
    rule: rule to considered
        rule 0 means that y0 is the largest among all output neurons.
    '''
    m = len(weight)
    n = len(bias)
    weight = np.array(weight)
    result = ""
    for i in range(n):
        output = ""
        if names[i] not in neuron:
            continue
        elif neuron[names[i]][1]:
            output += names[i] + " == True"
        else:
            output += names[i] + " == False"
        result += ("" if result == "" else " and ") + output
    if result != "":
        implication = get_implication_in_text(layerI + 1, rule)
        result += " => " + implication
        print(result)


def decision_tree_analysis(X, Y, feature_names):
    '''
    Outputs decision tree based of input X and output Y.

    Parameters:
    -----------
    X: decision tree input
    Y; decision tree output
    feature_names: names of all neurons in the inputs.
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    # Decision Tree
    decisionTree = DecisionTreeClassifier()
    decisionTree = decisionTree.fit(X_train, Y_train)

    Y_pred = decisionTree.predict(X_test)

    # print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    text_representation = tree.export_text(decisionTree, feature_names=feature_names)
    text_representation = text_representation.replace('<= 0.50', '== FALSE')
    text_representation = text_representation.replace('>  0.50', '== TRUE')
    print(text_representation)
    return decisionTree


def previous_layer_implication(weight, bias):
    '''
    Produce implications between each two consecutive layers.

    Parameters:
    -----------
    weight: weight of neural network
    bias: bias of neural network
    '''
    m = len(weight)
    n = len(bias)
    for layer in range(1, number_of_layer - 1):
        # imply from this layer to the next layer. 0 is ignored because of no true/false
        local_X = X[layer]
        if layer > 0:
            for case in local_X:
                for neuron in range(number_of_neurons_each_layer[layer]):
                    case[neuron] = case[neuron] > 0
        names = get_neuron_name_of_layer(layer)
        for rule in range(number_of_neurons_each_layer[layer + 1]):
            print("Layer: " + str(layer) + "\n" + "Rule: " + str(rule))
            Y = getY_each_layer_implication_activation(rule, X[layer + 1])
            if local_X == [] or Y == []:
                print("No properties")
                continue
            decisionTree = decision_tree_analysis(local_X, Y, names)
            traces = extract_decision_tree(decisionTree, names)
            if len(traces) == 0:
                print("No properties")
                continue
            # if layer > 0:
            for trace in traces:
                print(trace)
                print_implication_between_two_layers(weight[layer], bias[layer], trace, names, layer, rule)

        print()


def input_implication(weight, bias, neuron, feature_names):
    '''
    Outputs implication between inputs and ouputs in text

    Parameters:
    -----------
    weight: weight of the input layer
    bias: bias of input layer
    neuron: a map containing all neurons that are considered.
        Value = 0 means that it is FALSE or <= 0. Value = 1 means that it is TRUE or > 0.
    '''
    m = len(weight[0])
    n = len(bias)
    weight = np.array(weight)
    result = ""
    for i in range(n):
        output = ""
        for j in range(m):
            output += (" +" if weight[j][i] >= 0 else " ") + str(weight[j][i]) + "x" + str(j)
        if feature_names[i] not in neuron:
            continue
        elif neuron[feature_names[i]][1]:
            output += " > "
        else:
            output += " <= "
        output += str(float(-bias[j][0]) if bias[j][0] != 0 else bias[j][0])
        result += ("" if result == "" else " and ") + output
    implication = get_implication_in_text(number_of_layer, rule)
    result += " => " + implication
    print(result)


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


def checker_tool_non_input_layer():
    '''
    still developing not in used right now.
    '''
    X = [[] for i in range(number_of_layer + 1)]
    number_of_tests = 10
    for test in range(number_of_tests):
        inps = np.random.uniform(-10, 10, (1, number_of_neurons_each_layer[0]))
        inps.reshape(1, number_of_neurons_each_layer[0])
        # print(inps)
        # for layer in model.layers:
        #     keras_function = K.function([model.input], [layer.output])
        #     outputs.append(keras_function([inps, 1]))
        extractor = keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])
        outputs = extractor(inps)
        X[0].append(inps[0])
        cnt = 1
        for layer in outputs:
            X[cnt].append(layer.numpy()[0])
            cnt += 1

def checker_tool_input(model, weight, bias, number_of_layer, trace, names):
    '''
    Generate inputs according to some restriction to test proposed properties (still developing)
    Use Z3solver.

    :param model: model evaluating
    :param weight: weight of the first layer
    :param bias: bias of the first layer
    :param number_of_layer: number of layers in the neural network
    :param trace: proposed properties
    :param names: names of the neurons in that layers. Used to understand trace
    :return:
    True or False whether the properties are satisfied with randomly generated inputs. Not guaranteed that the properties are true.
    '''
    cntT = 0
    cntF = 0
    test_X = [[] for i in range(number_of_layer + 1)]
    number_of_tests = 10
    test = 0
    #inv_weight = np.array(weight).T
    print(weight)
    print(bias)
    inps = {}
    print(names)
    for i in range(len(bias)):
        print("i: " + str(i))
        variable = "x" + str(i)
        inps[variable] = z3.Real(variable)
    j = 0
    selected = {}
    print(selected)
    equation_list = []
    for name in names:
        if name in trace:
            z = z3.Real('z')
            z = bias[j][0]
            for i in range(len(bias)):
                variable = "x" + str(i)
                z = z + (inps[variable] * weight[i][j])
            z = z > 0 if trace[name][1] > 0 else z <= 0
            print(z)
            equation_list.append(z)
        j += 1
    print(equation_list)
    equations = [equation for equation in equation_list]
    s = z3.Solver()
    while test < number_of_tests:
        #equation = z3.And(equation, )
        s.add(equations)
        s.check()
        ans = s.model()
        for it in ans:
            print(it)
            print(type(it))
        #ans = z3.solve(equation)
        print(ans)
        exit(0)
        #inps = np.random.uniform(-10, 10, (1, number_of_neurons_each_layer[0]))
        #inps.reshape(1, number_of_neurons_each_layer[0])
        # print(inps)
        # for layer in model.layers:
        #     keras_function = K.function([model.input], [layer.output])
        #     outputs.append(keras_function([inps, 1]))
        extractor = keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])
        outputs = extractor(inps)
        print("here")
        print(trace)
        print(check_input_constraint(inps[0], weight, bias, trace))

        test_X[0].append(inps[0])

        cnt = 1
        for layer in outputs:
            test_X[cnt].append(layer.numpy()[0])
            cnt += 1
        test += 1
    print(cntT)
    print(cntF)


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
    '''
    onnx_model = onnx.load_model('../sample_input/eranmnist_benchmark/onnx/tf/mnist_relu_3_50.onnx')
    kera_model = onnx_to_keras(onnx_model, input_names=['0'])
    print(kera_model.summary())
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
            number_of_neurons_each_layer.append(np.array(layer_weight[0]).shape[0])
        number_of_neurons_each_layer.append(np.array(layer_weight[0]).shape[1])
        weight.append(np.array(layer_weight[0]))
        bias.append(np.array(layer_weight[1]))
    number_of_layer += 1
    print(number_of_layer)
    print(number_of_neurons_each_layer)
    print(weight)
    print(bias)
    return number_of_layer, number_of_neurons_each_layer, weight, bias


if __name__ == "__main__":

    # input_processing_onnx()
    # exit()

    model = Sequential()
    number_of_layer, number_of_neurons_each_layer, weight, bias, number_of_rule = 0, 0, [], [], 0
    filename = sys.argv[1]
    number_of_layer, number_of_neurons_each_layer, weight, bias = input_processing_json(filename)
    number_of_rule = number_of_neurons_each_layer[-1]

    import sys

    original_stdout = sys.stdout
    number_of_tests = 500
    with open(filename.replace('.json', '.txt').replace('input', 'output'), 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        activation = 'relu'
        for i in range(1, number_of_layer):
            # weight[i - 1] = np.array(weight[i - 1]).reshape(np.array(weight[i - 1]).shape)
            bias[i - 1] = np.array(bias[i - 1]).reshape(number_of_neurons_each_layer[i], 1)
            # print(weight[i - 1].shape)
            # print(bias[i - 1].shape)
            if i == 1:
                model.add(Dense(number_of_neurons_each_layer[i], input_shape=(number_of_neurons_each_layer[0],),
                                activation=activation,
                                kernel_initializer=tf.constant_initializer(weight[i - 1]),
                                bias_initializer=tf.constant_initializer(bias[i - 1]), dtype='float64'))
            elif i == number_of_layer - 1:
                model.add(Dense(number_of_neurons_each_layer[i],
                                kernel_initializer=tf.constant_initializer(weight[i - 1]),
                                bias_initializer=tf.constant_initializer(bias[i - 1]), dtype='float64'))
            else:
                model.add(Dense(number_of_neurons_each_layer[i], activation=activation,
                                kernel_initializer=tf.constant_initializer(weight[i - 1]),
                                bias_initializer=tf.constant_initializer(bias[i - 1]), dtype='float64'))
        model.summary()

        # print(weight)
        # print(bias)

        X = [[] for i in range(number_of_layer + 1)]

        for test in range(number_of_tests):
            inps = np.random.uniform(-10, 10, (1, number_of_neurons_each_layer[0]))
            inps.reshape(1, number_of_neurons_each_layer[0])
            # print(inps)
            # for layer in model.layers:
            #     keras_function = K.function([model.input], [layer.output])
            #     outputs.append(keras_function([inps, 1]))
            extractor = keras.Model(inputs=model.inputs,
                                    outputs=[layer.output for layer in model.layers])
            outputs = extractor(inps)
            X[0].append(inps[0])
            cnt = 1
            for layer in outputs:
                X[cnt].append(layer.numpy()[0])
                cnt += 1
        # checker_tool()
        # exit()
        previous_layer_implication(weight, bias)

        print("Each layer to the final output")
        for layer in range(1, number_of_layer - 1):
            print("Layer: " + str(layer))
            local_X = X[layer]
            names = get_neuron_name_of_layer(layer)
            for rule in range(number_of_neurons_each_layer[-1]):
                print("Rule: " + str(rule))
                Y = getY_implication_for_final_layer(rule, X[number_of_layer - 1])
                if local_X == [] or Y == []:
                    print("No properties.")
                    continue
                decisionTree = decision_tree_analysis(local_X, Y, names)
                traces = extract_decision_tree(decisionTree, names)
                if len(traces) == 0:
                    print("No properties")
                    continue
                for trace in traces:
                    if layer == 1:
                        input_implication(weight[0], bias[0], trace, names)
                    checker_tool_input(model, weight[0], bias[0], number_of_layer, trace, names)
                for trace in traces:
                    print_implication_between_two_layers(weight[layer], bias[layer], trace, names, number_of_layer - 1,
                                                         rule)
            print()
        sys.stdout = original_stdout  # Reset the standard output to its original value
        tf.saved_model.save(model, 'input_1')

    import keras2onnx

    onnx_model = keras2onnx.convert_keras(model, "test")
    filename_output_onnx = filename.replace('.json', '.onnx').replace('json', 'onnx')
    keras2onnx.save_model(onnx_model, filename_output_onnx)
