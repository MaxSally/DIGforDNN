from copy import deepcopy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split


NO_PROPERTY_MESSAGE = "No properties"

class NodePath:
    name = ""
    threshold = 0
    sign = 0

    def __init__(self, name, threshold, sign):
        self.name = name
        self.threshold = threshold
        self.sign = sign


class NeuronName:
    name = ""
    layer = -1
    order = -1

    def __init__(self, layer, order, number_of_layer):
        if layer == 0:
            NEURON = "x"
        elif layer == number_of_layer - 1:
            NEURON = "y"
        else:
            NEURON = "NEURON_" + str(layer)
        self.name = NEURON + "_" + str(order)

    def getName(self):
        return self.name

    def __str__(self):
        return self.name


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


def get_neuron_name_of_layer(layer, number_of_layer, number_of_neurons_each_layer):
    '''
    Outputs a list of neuron names of a layer

    Parameters:
    -----------
    layer: layer that user wants to retrieve names for
    '''
    names = []
    for i in range(number_of_neurons_each_layer[layer]):
        names.append(NeuronName(layer, i, number_of_layer).getName())
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


def get_implication_in_text(layer, number_of_layer, rule):
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


def print_implication_between_two_layers(weight, bias, number_of_layer, neuron, names, layerI, rule):
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
        implication = get_implication_in_text(layerI + 1, number_of_layer, rule)
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


def previous_layer_implication(X, weight, bias, number_of_layer, number_of_neurons_each_layer):
    '''
    Produce implications between each two consecutive layers.

    Parameters:
    -----------
    X: input for decisionTree
    weight: weight of neural network
    bias: bias of neural network
    number_of_layer: number of layer.
    number_of_neurons_each_layer: number of neurons at each layer
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
        names = get_neuron_name_of_layer(layer, number_of_layer, number_of_neurons_each_layer)
        for rule in range(number_of_neurons_each_layer[layer + 1]):
            print("Layer: " + str(layer) + "\n" + "Rule: " + str(rule))
            Y = getY_each_layer_implication_activation(rule, X[layer + 1])
            if local_X == [] or Y == []:
                print(NO_PROPERTY_MESSAGE)
                print("local_X: ", local_X)
                print("Y: ", Y)
                continue
            decisionTree = decision_tree_analysis(local_X, Y, names)
            traces = extract_decision_tree(decisionTree, names)
            print(traces)
            if len(traces) == 0:
                print(NO_PROPERTY_MESSAGE)
                continue
            # if layer > 0:
            for trace in traces:
                print(trace)
                print_implication_between_two_layers(weight[layer], bias[layer], number_of_layer, trace, names, layer, rule)

        print()


def input_implication(weight, bias, number_of_layer, neuron, feature_names, rule):
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
    implication = get_implication_in_text(number_of_layer, number_of_layer, rule)
    result += " => " + implication
    print(result)



