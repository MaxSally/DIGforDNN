import numpy as np
from tensorflow import keras
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import tree
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

tfds.disable_progress_bar()
tf.enable_v2_behavior()

(ds_train, ds_test), ds_info = tfds.load(
    'binary_alpha_digits',
    split=['train[20%:]', 'train[:80%]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_train = ds_train.batch(200)
ds_test = ds_test.batch(200)


# train_images = []
# train_labels = []
# test_images = []
# test_labels = []
#
# for images, labels in ds_train:
#     train_images.append(images.numpy())
#     train_labels.append(labels.numpy())
# for images, labels in ds_test:
#     test_images.append(images.numpy())
#     test_labels.append(labels.numpy())

model = tf.keras.models.load_model('binary_alpha_digit')
number_of_layer = 2
model.summary()
X = [[] for i in range(number_of_layer)]
Y = []

selection = 0

for i in range(1000):
    inps = ds_train
    print(inps)
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
