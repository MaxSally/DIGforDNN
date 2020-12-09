import sys
from copy import deepcopy
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import tree
from sklearn.model_selection import train_test_split  # Import train_test_split function
import json




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
        implication = ""
        NEURON_NAME = ("NEURON_" + str(layerI) if layerI < number_of_layer - 1 else "y")
        if rule == 0:
            implication = NEURON_NAME + "0 > " + NEURON_NAME + "1"
        elif rule == 1:
            implication = NEURON_NAME + "0 < " + NEURON_NAME + "1"
        elif rule == 2:
            implication = NEURON_NAME + "0 == " + NEURON_NAME + "1"
        elif rule == 3:
            implication = NEURON_NAME + "0 <= 0"
        else:
            implication = NEURON_NAME + "1 <= 0"
    result += " => " + implication
    print(result)


def get_name(layer):
    names = []
    NEURON = ""
    if layer < 0:
        NEURON = "x"
    elif layer == number_of_layer - 1:
        NEURON = "y"
    else:
        NEURON = "NEURON_" + str(layer)
    for i in range(number_of_neurons_each_layer[layer]):
        names.append(NEURON + str(i))
    return names


def getY(rule, Y):
    local_Y = []
    for example in Y:
        if rule == 0:
            local_Y.append(True if example[0] > example[1] else False)
        elif rule == 1:
            local_Y.append(True if example[0] < example[1] else False)
        elif rule == 2:
            local_Y.append(True if example[0] == example[1] else False)
        elif rule == 3:
            local_Y.append(True if example[0] <= 0 else False)
        else:
            local_Y.append(True if example[1] <= 0 else False)
    return local_Y


def decision_tree_analysis(X, Y, names):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    # Decision Tree
    decisionTree = DecisionTreeClassifier()
    decisionTree = decisionTree.fit(X_train, Y_train)

    Y_pred = decisionTree.predict(X_test)

    #print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    text_representation = tree.export_text(decisionTree, feature_names=names)
    text_representation = text_representation.replace('<= 0.50', '== FALSE')
    text_representation = text_representation.replace('>  0.50', '== TRUE')
    #print(text_representation)
    return decisionTree



def previous_layer_implication(weight, bias):
    m = len(weight)
    n = len(bias)
    for layer in range(1, number_of_layer):
        local_X = X[layer]
        if layer > 0:
            for case in local_X:
                for neuron in range(number_of_neurons_each_layer[layer]):
                    case[neuron] = case[neuron] > 0
        names = get_name(layer - 1)
        for rule in range(number_of_rule):
            print("Layer: " + str(layer) + "\n" + "Rule: " + str(rule))
            Y = getY(rule, X[layer + 1])
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
    implication = ""
    NEURON_NAME = "y"
    if rule == 0:
        implication = "y0 > y1"
    elif rule == 1:
        implication = "y0 < y1"
    elif rule == 2:
        implication = "y0 == y1"
    elif rule == 3:
        implication = "y0 <= 0"
    else:
        implication = "y1 <= 0"
    result += " => " + implication
    print(result)


def input_processing(filename):
    data = {}
    number_of_layer = -1
    number_of_neurons_each_layer = []
    number_of_rule = -1
    weight = []
    bias = []
    with open(filename, "r") as f:
        data = json.load(f)
        weight = data["weight"]
        bias = data["bias"]
        number_of_rule = data["number_of_rule"]
        number_of_neurons_each_layer = data["number_of_neurons_each_layer"]
        number_of_layer = data["number_of_layer"]
    return number_of_layer, number_of_neurons_each_layer, number_of_rule, weight, bias





if __name__ == "__main__":
    filename = sys.argv[1]
    model = Sequential()
    number_of_layer, number_of_neurons_each_layer, number_of_rule, weight, bias = input_processing(filename)

    # print(weight)
    # print(bias)
    activation = 'relu'
    for i in range(number_of_layer):
        if i == 0:
            model.add(Dense(number_of_neurons_each_layer[i], input_shape=(number_of_neurons_each_layer[i],), activation=activation,
                   kernel_initializer=tf.constant_initializer(weight[i]),
                   bias_initializer=tf.constant_initializer(bias[i]), dtype='float64'))
        elif i == number_of_layer - 1:
            model.add(Dense(number_of_neurons_each_layer[i],
                   kernel_initializer=tf.constant_initializer(weight[i]),
                   bias_initializer=tf.constant_initializer(bias[i]), dtype='float64'))
        else:
            model.add(Dense(number_of_neurons_each_layer[i], activation=activation,
                   kernel_initializer=tf.constant_initializer(weight[i]),
                   bias_initializer=tf.constant_initializer(bias[i]), dtype='float64'))
    model.summary()
    X = [[] for i in range(number_of_layer + 1)]

    for test in range(1000):
        inps = np.random.uniform(-100, 100, (1, number_of_neurons_each_layer[0]))
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
    for layer in range(1, number_of_layer):
        print("Layer: " + str(layer))
        local_X = X[layer]
        names = get_name(layer - 1)
        for rule in range(number_of_rule):
            print("Rule: " + str(rule))
            Y = getY(rule, X[number_of_layer])
            decisionTree = decision_tree_analysis(local_X, Y, names)
            traces = extract_decision_tree(decisionTree, names)
            if len(traces) == 0:
                print("No properties")
                continue
            for trace in traces:
                if layer == 1:
                    input_implication(weight[0], bias[0], trace, names)
            for trace in traces:
                layer_layer_implication(weight[layer], bias[layer], trace, names, number_of_layer, rule)
        print()

