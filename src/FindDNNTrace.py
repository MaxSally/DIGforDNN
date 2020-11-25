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
weight = np.array([[[1.0, 1.0], [-1.0, 1.0]], [[[0.5, -0.2], [-0.5, 0.1]]], [[[1.0, -1.0], [-1.0, 1.0]]]])
bias = np.array([[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]])

# print(weight)
# print(bias)
activation = 'relu'
predicate = "y1 > y2"
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

for test in range(1000):
    inps = np.random.uniform(-10, 10, (1, number_of_neurons_each_layer))
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
    for layer in outputs:
        for x in layer.numpy() > 0:
            # if(cnt == 0 and x[0] > 0 and x[1] > 0):
            #     print(inps)
            #     print(outputs)
            if cnt < len(outputs) - 1:
                X[cnt].append(x)
            else:
                tempY.append(True if x[0] > x[1] else False)
            cnt += 1
    Y.append(tempY)


# print(X)
# print(Y)

class NodePath:
    name = ""
    state = True

    def __init__(self, name, state):
        self.name = name
        self.state = state


def input_implication(weight, bias, neuron):
    m = len(weight)
    n = len(bias)
    weight = np.array(weight).T
    result = ""
    for i in range(n):
        output = ""
        for j in range(m):
            output += str(weight[i][j]) + ".x" + str(j) + " + "
        output += str(bias[j][0])
        if neuron[i]:
            output += " > 0"
        else:
            output += " <= 0"
        result += ("" if result == "" else " and ") + output
    result += " -> " + predicate
    print(result)

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
                path.append(NodePath(features[node], False))
                recurse(left, right, threshold, features, left[node], path,
                        depth + 1)
                if right[node] != -1:
                    originalPath.append(NodePath(features[node], True))
                    recurse(left, right, threshold, features, right[node],
                            originalPath,  depth + 1)
        else:
            if value[node][0][0] < value[node][0][1]:
                case = []
                for nodePath in path:
                    case.append(nodePath.state)
                result.append(case)

    recurse(left, right, threshold, features, 0, path)
    return result


for layer in range(number_of_layer - 1):
    X_train, X_test, Y_train, Y_test = train_test_split(X[layer], Y, test_size=0.3, random_state=1)

    # Decision Tree
    decisionTree = DecisionTreeClassifier()
    decisionTree = decisionTree.fit(X_train, Y_train)

    Y_pred = decisionTree.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    # (rules_list, values_path) = get_rules(decisionTree, X)
    # print(rules_list)
    names = []
    NEURON = "NEURON_"
    for i in range(number_of_neurons_each_layer):
        names.append(str(NEURON + str(layer) + str(i)))
    text_representation = tree.export_text(decisionTree, feature_names=names)
    text_representation = text_representation.replace('<= 0.50', '== FALSE')
    text_representation = text_representation.replace('>  0.50', '== TRUE')
    print(text_representation)
    if layer == 0:
        trace = extract_decision_tree(decisionTree, names)
        input_implication(weight[0], bias[0], trace[0])







