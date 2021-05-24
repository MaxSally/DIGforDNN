import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras

from Model import *
from CheckerTool import *
from Helper import *
import onnx
from onnx2keras import onnx_to_keras

if __name__ == "__main__":

    # input_processing_onnx()
    # exit()



    filename = sys.argv[1]
    input_mode = sys.argv[2]

    # createAndSaveModelAsOnnx(filename)
    # exit(0)

    number_of_layer, number_of_neurons_each_layer, weight, bias, number_of_rule = 0, [], [], [], 0
    if input_mode == 'json':
        number_of_layer, number_of_neurons_each_layer, weight, bias = input_processing_json(filename)
    elif input_mode == 'onnx':
        number_of_layer, number_of_neurons_each_layer, weight, bias = input_processing_onnx(filename)
    number_of_rule = number_of_neurons_each_layer[-1]
    model = createModel(number_of_layer, number_of_neurons_each_layer, weight, bias)GI

    original_stdout = sys.stdout
    number_of_tests = 500

    with open(filename.replace('.json', '.txt').replace('input', 'output'), 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.


        X = [[] for i in range(number_of_layer)]

        for test in range(number_of_tests):
            inps = np.random.uniform(-10, 10, (1, number_of_neurons_each_layer[0]))
            # inps.reshape(1, number_of_neurons_each_layer[0])
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
        previous_layer_implication(X, weight, bias, number_of_layer, number_of_neurons_each_layer)

        print("Each layer to the final output")
        for layer in range(1, number_of_layer - 1):
            print("Layer: " + str(layer))
            local_X = X[layer]
            names = get_neuron_name_of_layer(layer, number_of_layer, number_of_neurons_each_layer)
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
                        input_implication(weight[0], bias[0], number_of_layer, trace, names, rule)
                    #print(trace)
                    checker_tool_input(model, weight[0], bias[0], number_of_layer, number_of_neurons_each_layer, rule, names, trace)
                for trace in traces:
                    print_implication_between_two_layers(weight[layer], bias[layer], number_of_layer, trace, names, number_of_layer - 1,
                                                         rule)
            print()
        sys.stdout = original_stdout  # Reset the standard output to its original value
        tf.saved_model.save(model, filename.replace('.json', ''))

    saveModelAsOnnx(model, filename.replace('.json', '.onnx').replace('json', 'onnx'))


