import sys
from copy import deepcopy
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


class NodePath:
    name = ""
    threshold = 0
    sign = 0

    def __init__(self, name, threshold, sign):
        self.name = name
        self.threshold = threshold
        self.sign = sign


def printDT(decisionTree, names):
    text_representation = tree.export_text(decisionTree, feature_names=names)
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

    left = tree.tree_.children_left
    right = tree.tree_.children_right
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

    recurse(left, right, threshold, features, 0, path)
    return result


def get_name(layer):
    names = []
    NEURON = ""
    if layer == 0:
        NEURON = "x"
    elif layer == number_of_layer - 1:
        NEURON = "y"
    else:
        NEURON = "NEURON_" + str(layer)
    for i in range(number_of_neurons_each_layer[layer]):
        names.append(NEURON + "_" +  str(i))
    return names


def getY_final_layer_implication(rule, Y):
    local_Y = []
    for example in Y:
        local_Y.append(example[rule] > max([x for i, x in enumerate(example) if i != rule]))
    return local_Y

def getY_each_layer_implication_activation(rule, Y):
    local_Y = []
    for example in Y:
        local_Y.append(example[rule] > 0)
    return local_Y


def get_implication(layer, rule):
    implication = ""
    NEURON_NAME = ("NEURON_" + str(layer) if layer < number_of_layer - 1 else "y")
    implication = NEURON_NAME + "_" + str(rule)
    return implication


def layer_layer_implication(weight, bias, neuron, names, layerI, rule):
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
        implication = get_implication(layerI + 1, rule)
        result += " => " + implication
        print(result)


def decision_tree_analysis(X, Y, names):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    # Decision Tree
    decisionTree = DecisionTreeClassifier()
    decisionTree = decisionTree.fit(X_train, Y_train)

    Y_pred = decisionTree.predict(X_test)

    # print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    text_representation = tree.export_text(decisionTree, feature_names=names)
    text_representation = text_representation.replace('<= 0.50', '== FALSE')
    text_representation = text_representation.replace('>  0.50', '== TRUE')
    # print(text_representation)
    return decisionTree


def previous_layer_implication(weight, bias):
    m = len(weight)
    n = len(bias)
    for layer in range(1, number_of_layer - 1):
        #imply from this layer to the next layer. 0 is ignored because of no true/false
        local_X = X[layer]
        if layer > 0:
            for case in local_X:
                for neuron in range(number_of_neurons_each_layer[layer]):
                    case[neuron] = case[neuron] > 0
        names = get_name(layer)
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
                layer_layer_implication(weight[layer], bias[layer], trace, names, layer, rule)
        print()


def input_implication(weight, bias, neuron, names):
    m = len(weight)
    n = len(bias)
    weight = np.array(weight)
    result = ""
    for i in range(n):
        output = ""
        for j in range(m):
            output += str(weight[i][j]) + ".x" + str(j) + " + "
        output += str(bias[j][0])
        if names[i] not in neuron:
            continue
        elif neuron[names[i]][1]:
            output += " > 0"
        else:
            output += " <= 0"
        result += ("" if result == "" else " and ") + output
    implication = get_implication(number_of_layer, rule)
    result += " => " + implication
    print(result)


def input_processing(filename):
    data = {}
    number_of_layer = -1
    number_of_neurons_each_layer = []
    weight = []
    bias = []
    with open(filename, "r") as f:
        data = json.load(f)
        weight = data["weight"]
        bias = data["bias"]
        number_of_neurons_each_layer = data["number_of_neurons_each_layer"]
        number_of_layer = data["number_of_layer"]
    return number_of_layer, number_of_neurons_each_layer, weight, bias


if __name__ == "__main__":

    model = Sequential()
    number_of_layer, number_of_neurons_each_layer, weight, bias, number_of_rule = 0, 0, [], [], 0
    filename = sys.argv[1]
    number_of_layer, number_of_neurons_each_layer, weight, bias = input_processing(filename)
    number_of_rule = number_of_neurons_each_layer[-1]

    import sys
    original_stdout = sys.stdout
    with open(filename.replace('.json', '.txt').replace('input', 'output'), 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        activation = 'relu'
        for i in range(1, number_of_layer):
            #weight[i - 1] = np.array(weight[i - 1]).reshape(np.array(weight[i - 1]).shape)
            bias[i - 1] = np.array(bias[i - 1]).reshape(number_of_neurons_each_layer[i], 1)
            #print(weight[i - 1].shape)
            #print(bias[i - 1].shape)
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

        for test in range(1000):
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
                for node in layer.numpy():
                    X[cnt].append(node)
                    cnt += 1
        previous_layer_implication(weight, bias)

        print("Each layer to the final output")
        for layer in range(1, number_of_layer - 1):
            print("Layer: " + str(layer))
            local_X = X[layer]
            names = get_name(layer)
            for rule in range(number_of_neurons_each_layer[-1]):
                print("Rule: " + str(rule))
                Y = getY_final_layer_implication(rule, X[number_of_layer - 1])
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
                for trace in traces:
                    layer_layer_implication(weight[layer], bias[layer], trace, names, number_of_layer - 1, rule)
            print()
        sys.stdout = original_stdout  # Reset the standard output to its original value
        #model.save('input_3.pb')
