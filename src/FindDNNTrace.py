import random
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
number_of_layer = 2
weight = np.array([[[1.0, 1.0], [-1.0, 1.0]], [[[0.5, -0.2], [-0.5, 0.1]]], [[[1.0, -1.0], [-1.0, 1.0]]]])
bias = np.array([[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]])
activation = 'relu'
layer1 = Dense(2, input_shape=(2,), activation=activation, kernel_initializer=tf.constant_initializer(weight[0]),
               bias_initializer=tf.constant_initializer(bias[0]))
layer2 = Dense(2, activation=activation, kernel_initializer=tf.constant_initializer(weight[1]),
               bias_initializer=tf.constant_initializer(bias[1]))
layer3 = Dense(2, kernel_initializer=tf.constant_initializer(weight[2]),
               bias_initializer=tf.constant_initializer(bias[2]))
model.add(layer1)
model.add(layer2)
model.add(layer3)
X = [[] for i in range(number_of_layer)]
Y = []

for test in range(60):
    a = random.randint(-10, 10)
    b = random.randint(-10, 10)
    inps = np.array([[a, b]])
    inps.reshape(1, 2)
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

            if cnt < len(outputs) - 1:
                X[cnt].append(x)
            else:
                tempY.append(True if x[0] > x[1] else False)
            cnt += 1
    Y.append(tempY)
print(X)
print(Y)

def get_rules(dtc, df):
    rules_list = []
    values_path = []
    values = dtc.tree_.value

    def RevTraverseTree(tree, node, rules, pathValues):
        '''
        Traverase an skl decision tree from a node (presumably a leaf node)
        up to the top, building the decision rules. The rules should be
        input as an empty list, which will be modified in place. The result
        is a nested list of tuples: (feature, direction (left=-1), threshold).
        The "tree" is a nested list of simplified tree attributes:
        [split feature, split threshold, left node, right node]
        '''
        # now find the node as either a left or right child of something
        # first try to find it as a left node

        try:
            prevnode = tree[2].index(node)
            leftright = '<='
            pathValues.append(values[prevnode])
        except ValueError:
            prevnode = tree[3].index(node)
            leftright = '>'
            pathValues.append(values[prevnode])

        # p1 = df.columns[tree[0][prevnode]]
        p1 = tree[0][prevnode]
        p2 = tree[1][prevnode]
        rules.append('Neuron ' + str(p1) + ' ' + leftright + ' ' + str(p2))

        if prevnode != 0:
            RevTraverseTree(tree, prevnode, rules, pathValues)

    leaves = dtc.tree_.children_left == -1
    leaves = np.arange(0, dtc.tree_.node_count)[leaves]

    thistree = [dtc.tree_.feature.tolist()]
    thistree.append(dtc.tree_.threshold.tolist())
    thistree.append(dtc.tree_.children_left.tolist())
    thistree.append(dtc.tree_.children_right.tolist())

    # get the decision rules for each leaf node & apply them
    for (ind, nod) in enumerate(leaves):
        # get the decision rules
        rules = []
        pathValues = []
        RevTraverseTree(thistree, nod, rules, pathValues)

        pathValues.insert(0, values[nod])
        pathValues = list(reversed(pathValues))

        rules = list(reversed(rules))

        rules_list.append(rules)
        values_path.append(pathValues)

    return (rules_list, values_path)

for layer in range(number_of_layer):
    X_train, X_test, Y_train, Y_test = train_test_split(X[layer], Y, test_size=0.3, random_state=1)

    # Decision Tree
    decisionTree = DecisionTreeClassifier()
    decisionTree = decisionTree.fit(X_train, Y_train)

    Y_pred = decisionTree.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

    (rules_list, values_path) = get_rules(decisionTree, X)
    print(rules_list)

    text_representation = tree.export_text(decisionTree)
    print(text_representation)
