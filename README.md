1. Requirements

Library required: Tensorflow 2.3.1, Numpy, sklearn (DecisionTree), Copy, and z3.
Main file is 'FindDnnTrace.py'.

2. Running:
- Make sure all the required libraries are installed in the Python interpreter. Run `FindDNNTrace.py`.
    + If needed, downgrading to `python 3.6` for conda environment if use virtual environment.  
-   Command line: 
    + Go to `DIGforDNN/src`
    + `chmod +x FindDNNTrace.py` to allow execution on the program
    + `python FindDNNTrace.py INPUT_FILE RUNNING_FILE_TYPE` to run it. Expected output shown in section 3.
    + Sample input is `input.json`. As of right now, the program only accepts input in json format. 
    + If needed to install library: `python -m pip install PACKAGE_NAME`

3. Example:
- Let say we want to find property of model 1 (stored in `sample_input/json/sample_input_1.json` or `sample_input/onnx/sample_input_1.onnx`).
- Use command `python FindDNNTrace.py ../sample_input/json/sample_input_1.json json`
- The output will get written to `sample_output/json/sample_output_1.txt` or `sample_output/onnx/sample_output_1.txt` depend on input parameter.
- An abridged version of output:
```Layer: 1
Rule: 0
NEURON_00 == True and NEURON_01 == False => NEURON_10
Layer: 1
Rule: 1
NEURON_00 == False and NEURON_01 == True => NEURON_11

Layer: 2
Rule: 0
NEURON_10 == True => y0
Layer: 2
Rule: 1
NEURON_11 == True => y1

Each layer to the final output
Layer: 1
Rule: 0
1.0.x0 + -1.0.x1 + 0.0 > 0 and 1.0.x0 + 1.0.x1 + 0.0 <= 0 => y0
NEURON_00 == True and NEURON_01 == False => y0
Rule: 1
1.0.x0 + -1.0.x1 + 0.0 <= 0 and 1.0.x0 + 1.0.x1 + 0.0 > 0 => y1
NEURON_00 == False and NEURON_01 == True => y1

Layer: 2
Rule: 0
NEURON_10 == True => y0
Rule: 1
NEURON_11 == True => y1

```

- Noted that this sample output does not include all the actual output. The actual output contains some information regarding the testing process that is under development. More information can be foudn below.
- Input interpretation: the conclusion of each property (the part after =>) is a single variable indicating that among similar type, that variable is the largest in value. E.g:`1.0.x0 + -1.0.x1 + 0.0 > 0 and 1.0.x0 + 1.0.x1 + 0.0 <= 0 => y0` means that if `1.0.x0 + -1.0.x1 + 0.0 > 0 and 1.0.x0 + 1.0.x1 + 0.0 <= 0` is true then `y0` (y means it is the output neuron) is larger than any other output neurons. `y0 > max(y_1, y_2, ....y_n)`.
4. Checking tool:
- As of right now (May 4th), we have not found a nice way to check our current output using existing solver. Therefore, we use Z3Solver to do some checkings.
- First, Z3 Solver tried to find a combination of input value for neurons in input layer that satisfies the if part of the proposed property (property found by the program). Then, the program uses those values to see the conclusions of the property are also satisfied.
- E.g: in the sample problem, the proposed property is `1.0.x0 + -1.0.x1 + 0.0 > 0 and 1.0.x0 + 1.0.x1 + 0.0 <= 0 => y0`. The testing first finds sets of `x0, x1` such that `1.0.x0 + -1.0.x1 + 0.0 > 0 and 1.0.x0 + 1.0.x1 + 0.0` is true. Then the program determines whether the output neuron `y0` is larger than any other output neurons. In this case, since there are only 2 output neurons, the conclusion is satisfied as long as `y0> y1`. 
- This is a really limiting tool as it does not prove every single case. A solver is still much more desirable.

5. Future features and road path:

a. Better output: The mathematical expression can be improved. It has been done as parts of the checking tool features, so it can be easily transfered (not high priority).

b. More information extraction: The decision tree still has more information waiting to be extracted. E.g: it sometimes catches rare cases. However, those cases usually result in problems, so I have place checkers to prevent them from giving code errors. However, they may be a desirable studying objects as those cases are usually interesting. 

c. Properties of large neural networks await verification from reliable tool (E.g DNNV, Maribou). Their system is not built for this type of property (implication) yet, so hopefully we could find an alternative in the future. 


