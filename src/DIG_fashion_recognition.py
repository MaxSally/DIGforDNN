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

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.load_model('fashion_recognition.h5')
number_of_layer = 3
model.summary()
X = [[] for i in range(number_of_layer)]
Y = []

selection = 0

for i in range(1000):  # len(train_images)
    inps = train_images[i:i + 1]
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    outputs = extractor(inps)
    # print(outputs)
    tempY = []
    cnt = 0
    for layer in outputs:
        for x in layer.numpy():
            if cnt < len(outputs) - 1:
                X[cnt].append(x)
            else:
                tempY.append(True if np.where(x==max(x)) == selection else False)
            cnt += 1
    Y.append(tempY)

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
    for i in range(len(X_train[0])):
        names.append(str(NEURON + str(layer) + str(i)))
    text_representation = tree.export_text(decisionTree, feature_names=names)
    print(text_representation)
