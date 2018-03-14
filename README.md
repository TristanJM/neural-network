# neural-network
Multi Layer Perceptron neural network with no ML libraries. Uses Genetic Algorithms for optimisation. Written in Python.

* Model has 1 hidden layer with arbitrary neurons.
* Model output layer has 1 neuron.
* Trains with back propagation using gradient descent of the error function.
* Model optimisation is done via a Genetic Evolutionary Algorithm

## Installation
```
pip install numpy pandas copy matplotlib warnings deap scipy bitstring
```

## Results
The best results for the river basin's pan evaporation prediction (Fresno, CA):

|Hidden neurons|8|
|Learning rate|0.1|
|Data split|60/20/20|
|Momentum|True (alpha = 1.1)|
|Bold Driver|False|
|Random seed|10|
|Validation RMSE|0.010292|
|Test RMSE|0.009849|
|Denormalised RMSE|0.158563|

Plotting prediction:
![Pan evaporation plotted prediction](/results/PanE Prediction.png "Pan evaporation prediction")
