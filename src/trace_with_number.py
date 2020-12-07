import random
from copy import deepcopy
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import tree
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt

model = Sequential()
number_of_layer = 3
number_of_neurons_each_layer = 2
number_of_rule = 5
weight = np.array([[[1.0, -1.0], [1.0, 1.0]], [[0.5, -0.2], [-0.5, 0.1]], [[1.0, -1.0], [-1.0, 1.0]]])
bias = np.array([[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]])

# print(weight)
# print(bias)
activation = 'relu'
queryType = 0
predicate = ["y1 > y2", "y1 < y2", "y1 == y2", "y1 <= 0", "y2 <= 0"]
print("Predicate: " + predicate[queryType])
layer1 = Dense(number_of_neurons_each_layer, input_shape=(number_of_neurons_each_layer,), activation=activation,
               kernel_initializer=tf.constant_initializer(weight[0]),
               bias_initializer=tf.constant_initializer(bias[0]), dtype='float64')
layer2 = Dense(number_of_neurons_each_layer, activation=activation,
               kernel_initializer=tf.constant_initializer(weight[1]),
               bias_initializer=tf.constant_initializer(bias[1]), dtype='float64')
layer3 = Dense(number_of_neurons_each_layer, kernel_initializer=tf.constant_initializer(weight[2]),
               bias_initializer=tf.constant_initializer(bias[2]), dtype='float64')

model.add(layer1)
model.add(layer2)
model.add(layer3)
X = [[] for i in range(number_of_layer)]
Y = []

for test in range(100):
    inps = np.random.uniform(-100, 100, (1, number_of_neurons_each_layer))
    inps.reshape(1, number_of_neurons_each_layer)
    # print(inps)
    # for layer in model.layers:
    #     keras_function = K.function([model.input], [layer.output])
    #     outputs.append(keras_function([inps, 1]))
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    outputs = extractor(inps)
    tempY = []
    cnt = 0
    X[cnt].append(inps[0])
    cnt = 1
    for layer in outputs:
        for node in layer.numpy():
            if cnt < len(outputs):
                X[cnt].append(node)
            else:
                node = node > 0
                if queryType == 0:
                    tempY.append(True if node[0] > node[1] else False)
                elif queryType == 1:
                    tempY.append(True if node[0] < node[1] else False)
                elif queryType == 2:
                    tempY.append(True if node[0] == node[1] else False)
                elif queryType == 3:
                    tempY.append(True if node[0] <= 0 else False)
                else:
                    tempY.append(True if node[1] <= 0 else False)
            cnt += 1
    Y.append(tempY)

class NodePath:
    name = ""
    threshold = 0
    sign = 0

    def __init__(self, name, threshold, sign):
        self.name = name
        self.threshold = threshold
        self.sign = sign




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
            if value[node][0][0] < value[node][0][1] and value[node][0][0] == 0:
                case = dict()
                for nodePath in path:
                    case[nodePath.name] = [round(nodePath.threshold, 2), nodePath.sign]
                result.append(case)

    recurse(left, right, threshold, features, 0, path)
    return result


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
            output += " > " + str(neuron[names[i]][0])
        else:
            output += " <= " + str(neuron[names[i]][0])
        result += ("" if result == "" else " and ") + output
    result += " -> " + predicate[queryType]
    print(result)

def consecutive_implication(weight, bias, neuron, names, layerI, rule):
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
        if rule == 0:
            implication = "=> NEURON_" + str(layerI) + "0 > NEURON_" + str(layerI) + "1"
        elif rule == 1:
            implication = "=> NEURON_" + str(layerI) + "0 < NEURON_" + str(layerI) + "1"
        elif rule == 2:
            implication = "=> NEURON_" + str(layerI) + "0 == NEURON_" + str(layerI) + "1"
        elif rule == 3:
            implication = "=> NEURON_" + str(layerI) + "0 < 0"
        else:
            implication = "=> NEURON_" + str(layerI) + "1 < 0"
    result += " " + implication
    print(result)

def get_name(layer):
    names = []
    NEURON = "NEURON_"
    for i in range(number_of_neurons_each_layer):
        names.append((str(NEURON + str(layer) if layer >= 0 else "x") + str(i)))
    return names

def previous_layer_implication(weight, bias):
    m = len(weight)
    n = len(bias)
    for layer in range(number_of_layer):
        local_X = X[layer]
        if layer > 0:
            for case in local_X:
                for neuron in range(number_of_neurons_each_layer):
                    case[neuron] = case[neuron] > 0
        if layer == number_of_layer - 1:
            break
        names = get_name(layer - 1)
        for rule in range(number_of_rule):
            print("Layer: " + str(layer) + "\n" + "Rule: " + str(rule))
            local_Y = []
            for example in X[layer + 1]:
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
            X_train, X_test, Y_train, Y_test = train_test_split(local_X, local_Y, test_size=0.3, random_state=1)
            # Decision Tree
            decisionTree = DecisionTreeClassifier()
            decisionTree = decisionTree.fit(X_train, Y_train)

            Y_pred = decisionTree.predict(X_test)

            print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
            text_representation = tree.export_text(decisionTree, feature_names=names)
            text_representation = text_representation.replace('<= 0.50', '== FALSE')
            text_representation = text_representation.replace('>  0.50', '== TRUE')
            print(text_representation)
            traces = extract_decision_tree(decisionTree, names)
            #if layer > 0:
            for trace in traces:
                consecutive_implication(weight[layer], bias[layer], trace, names, layer, rule)


previous_layer_implication(weight, bias)

for layer in range(1, number_of_layer):
    X_train, X_test, Y_train, Y_test = train_test_split(X[layer], Y, test_size=0.3, random_state=1)
    print("Layer: " + str(layer))
    # Decision Tree
    names = get_name(layer - 1)
    decisionTree = DecisionTreeClassifier()
    decisionTree = decisionTree.fit(X_train, Y_train)

    Y_pred = decisionTree.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    # (rules_list, values_path) = get_rules(decisionTree, X)
    # print(rules_list)

    text_representation = tree.export_text(decisionTree, feature_names=names)
    text_representation = text_representation.replace('<= 0.50', '== FALSE')
    text_representation = text_representation.replace('>  0.50', '== TRUE')
    print(text_representation)
    traces = extract_decision_tree(decisionTree, names)
    for trace in traces:
        if layer == 0:
            input_implication(weight[0], bias[0], trace, names)

