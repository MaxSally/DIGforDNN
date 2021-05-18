import numpy as np
from tensorflow import keras
import z3

def checker_tool_non_input_layer():
    '''
    still developing not in used right now.
    '''
    X = [[] for i in range(number_of_layer + 1)]
    number_of_tests = 100
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


def checker_tool_input(model, weight, bias, number_of_layer, number_of_neurons_each_layer, rule, names, trace):
    '''
    Generate inputs according to some restriction to test proposed properties (still developing)
    Use Z3solver.

    :param model: model evaluating
    :param weight: weight of the first layer
    :param bias: bias of the first layer
    :param number_of_layer: number of layers in the neural network
    :param number_of_neurons_each_layer: a list of number of neurons of each layer
    :param rule: proposed properties
    :param names: names of the neurons in that layers. Used to understand trace
    :param trace: decision tree output trace
    :return:
    True or False whether the properties are satisfied with randomly generated inputs. Not guaranteed that the properties are true.
    '''
    cntT = 0
    cntF = 0
    test_X = [[] for i in range(number_of_layer)]
    number_of_tests = 100
    test = 0
    inputs_variables_name = {}
    for i in range(len(bias)):
        variable = "x" + str(i)
        inputs_variables_name[variable] = z3.Real(variable)
    j = 0
    selected = {}
    equation_list = []
    for name in names:
        if name in trace:
            z = z3.Real('z')
            z = bias[j][0]
            for i in range(len(bias)):
                variable = "x" + str(i)
                z = z + (inputs_variables_name[variable] * weight[i][j])
            z = z > 0 if trace[name][1] > 0 else z <= 0
            equation_list.append(z)
        j += 1
    equations = [equation for equation in equation_list]
    s = z3.Solver()
    s.add(equations)
    while test < number_of_tests:
        s.check()
        ans = s.model()
        test_inputs = [0] * number_of_neurons_each_layer[0]
        t = z3.Solver()
        assignment_list = []
        for input_assignment in ans:
            input_assignment_str = str(input_assignment)
            assignment_list.append(inputs_variables_name[input_assignment_str] == ans[input_assignment])
            index = int(input_assignment_str[1:])
            test_inputs[index] = float(ans[input_assignment].numerator_as_long()) / float(
                ans[input_assignment].denominator_as_long())
        t.add(z3.Not(z3.And(assignment_list)))
        s.add(t.assertions())
        test_inputs = np.array(test_inputs).reshape((1, number_of_neurons_each_layer[0]))
        extractor = keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])
        outputs = extractor(test_inputs)
        test_X[0].append(test_inputs[0])
        cnt = 1
        for layer in outputs:
            test_X[cnt].append(layer.numpy()[0])
            cnt += 1
        test += 1
        last_layer = test_X[-1][0]

        if len(last_layer) < 2:
            print("Last layer does not have enough nodes")
            return
        maxValue = last_layer[0] if rule != 0 else last_layer[1]
        for i in range(len(last_layer)):
            if rule == i:
                continue
            maxValue = max(maxValue, last_layer[i])
        if last_layer[rule] > maxValue:
            cntT += 1
        else:
            cntF += 1
    print("True count: " + str(cntT) + " out of " + str(number_of_tests))
    print("False count: " + str(cntF) + " out of " + str(number_of_tests))


def checker_tool_input_2(model, weight, bias, number_of_layer, number_of_neurons_each_layer, rule, names, trace, varying_neuron):
    '''
    Generate inputs according to some restriction to test proposed properties (still developing)
    Use Z3solver.
    Fix all variables but 1.
    :param model: model evaluating
    :param weight: weight of the first layer
    :param bias: bias of the first layer
    :param number_of_layer: number of layers in the neural network
    :param number_of_neurons_each_layer: a list of number of neurons of each layer
    :param rule: proposed properties
    :param names: names of the neurons in that layers. Used to understand trace
    :param trace: decision tree output trace
    :param varying_neuron: neurons that you wish to fix value.
    :return:
    True or False whether the properties are satisfied with randomly generated inputs. Not guaranteed that the properties are true.
    '''
    cntT = 0
    cntF = 0
    test_X = [[] for i in range(number_of_layer)]
    number_of_tests = 20
    test = 0
    inputs_variables_name = {}
    for i in range(len(bias)):
        variable = "x" + str(i)
        inputs_variables_name[variable] = z3.Int(variable)
    j = 0
    selected = {}
    equation_list = []
    for name in names:
        if name in trace:
            z = z3.Real('z')
            z = bias[j][0]
            for i in range(len(bias)):
                variable = "x" + str(i)
                z = z + (inputs_variables_name[variable] * weight[i][j])
            z = z > 0 if trace[name][1] > 0 else z <= 0
            equation_list.append(z)
        j += 1
    equations = [equation for equation in equation_list]
    s = z3.Solver()
    s.add(equations)
    s.push()
    first_run = True
    initialFixAssigment = []
    countTestEachFixAssignment = 0
    while test < number_of_tests:
        endingTests = False
        while s.check().__str__() == 'unsat':
            if not initialFixAssigment:
                print('No more assignments in general. Ending tests')
                print(s)
                endingTests = True
                break
            print('No more assignments for current fix assignment. Changing to new fix values')
            s.pop()
            s.add(z3.Not(z3.And(initialFixAssigment)))
            s.push()
            initialFixAssigment = []
            print(s)
            first_run = True

        if endingTests:
            break
        ans = s.model()
        countTestEachFixAssignment += 1
        test_inputs = [0] * number_of_neurons_each_layer[0]
        print(ans)
        print(s)
        t = z3.Solver()
        assignment_list = []
        for input_assignment in ans:
            input_assignment_str = str(input_assignment)
            index = int(input_assignment_str[1:])
            test_inputs[index] = ans[input_assignment].as_long()
            if first_run and index != varying_neuron:
                assignment_list.append(inputs_variables_name[input_assignment_str] == ans[input_assignment])
                initialFixAssigment.append(inputs_variables_name[input_assignment_str] == ans[input_assignment])
            if index == varying_neuron:
                assignment_list.append(inputs_variables_name[input_assignment_str] != ans[input_assignment])

        if first_run:
            first_run = False

        t.add(z3.And(assignment_list))
        s.add(t.assertions())

        test_inputs = np.array(test_inputs).reshape((1, number_of_neurons_each_layer[0]))
        extractor = keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])
        outputs = extractor(test_inputs)
        test_X[0].append(test_inputs[0])
        cnt = 1
        for layer in outputs:
            test_X[cnt].append(layer.numpy()[0])
            cnt += 1
        test += 1
        last_layer = test_X[-1][0]

        # print("Compare outputs")
        # print(last_layer)
        if len(last_layer) < 2:
            print("Last layer does not have enough nodes")
            continue
        maxValue = last_layer[0] if rule != 0 else last_layer[1]
        for i in range(len(last_layer)):
            if rule == i:
                continue
            maxValue = max(maxValue, last_layer[i])
        if last_layer[rule] > maxValue:
            cntT += 1
        else:
            cntF += 1
    print("True count: " + str(cntT) + " out of " + str(number_of_tests))
    print("False count: " + str(cntF) + " out of " + str(number_of_tests))
    print()
