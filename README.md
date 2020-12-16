1. Requirements

Library required: Tensorflow 2.3.1, Numpy, sklearn (DecisionTree), Copy
Main file is 'FindDnnTrace.py'

2. Running:
- In Pycharm, make sure all the required libraries are installed in the Python interpreter. Run `FindDNNTrace.py`.
    + If needed, downgrading to `python 3.6` for conda environment.    
-   Command line: 
    + Go to `DIGforDNN/src`
    + `chmod +x FindDNNTrace.py` to allow execution on the program
    + `python FindDNNTrace.py INPUT_FILE` to run it. Expected output shown in section 3.
    + Sample input is `input.json`. As of right now, the program only accepts input in json format. 
    + If needed to install library: `python -m pip install PACKAGE_NAME`

3. Expected Output:

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
