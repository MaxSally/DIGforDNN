1. Requirements

Library required: Tensorflow 2.3.1, Numpy, sklearn (DecisionTree), Copy
Main file is 'FindDnnTrace.py'

2. Running:
- In Pycharm, make sure all the required libraries are installed in the Python interpreter. Run `FindDNNTrace.py`.
    + If needed, downgrading to `python 3.6` for conda environment.    
- Command line: 
    -`chmod +x FindDNNTrace.py`
    - `python FindDNNTrace.py`
    - If need to install library: `python -m pip install PACKAGE_NAME`

3. Expected Output:

```Layer: 1
Rule: 0
NEURON_00 == True and NEURON_01 == False => NEURON_10 > NEURON_11
Layer: 1
Rule: 1
NEURON_00 == False and NEURON_01 == True => NEURON_10 < NEURON_11
Layer: 1
Rule: 2
NEURON_00 == False and NEURON_01 == False => NEURON_10 == NEURON_11
Layer: 1
Rule: 3
NEURON_00 == False => NEURON_10 < 0
Layer: 1
Rule: 4
NEURON_01 == False => NEURON_11 < 0
Layer: 2
Rule: 0
NEURON_10 == True => NEURON_20 > NEURON_21
Layer: 2
Rule: 1
NEURON_11 == True => NEURON_20 < NEURON_21
Layer: 2
Rule: 2
NEURON_10 == False and NEURON_11 == False => NEURON_20 == NEURON_21
Layer: 2
Rule: 3
NEURON_10 == False => NEURON_20 < 0
Layer: 2
Rule: 4
NEURON_11 == False => NEURON_21 < 0
Each layer to the final output
Layer: 1
1.0.x0 + -1.0.x1 + 0.0 > 0 and 1.0.x0 + 1.0.x1 + 0.0 <= 0 => y0 > y1
NEURON_00 == True and NEURON_01 == False => y0 > y1
1.0.x0 + -1.0.x1 + 0.0 <= 0 and 1.0.x0 + 1.0.x1 + 0.0 > 0 => y0 < y1
NEURON_00 == False and NEURON_01 == True => y0 < y1
1.0.x0 + -1.0.x1 + 0.0 <= 0 and 1.0.x0 + 1.0.x1 + 0.0 <= 0 => y0 == y1
NEURON_00 == False and NEURON_01 == False => y0 == y1
1.0.x0 + -1.0.x1 + 0.0 <= 0 => y0 < 0
NEURON_00 == False => y0 < 0
1.0.x0 + 1.0.x1 + 0.0 <= 0 => y1 < 0
NEURON_01 == False => y1 < 0
Layer: 2
NEURON_10 == True => y0 > y1
NEURON_11 == True => y0 < y1
NEURON_10 == False and NEURON_11 == False => y0 == y1
NEURON_10 == False => y0 < 0
NEURON_11 == False => y1 < 0
```
