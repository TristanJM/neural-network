import numpy as np
import modules.const as const

np.random.seed(const.RANDOM_SEED)

# NN Model - Neuron Layer
class Layer:
    def __init__(self, num_inputs, num_neurons, activation_method):
        self.inputs = num_inputs
        self.neurons = num_neurons
        self.activation_method = activation_method

        # randomly assign neuron starting weights and bias (-2/inputs -> 2/inputs)
        self.weights = np.random.normal(0, float(2)/self.inputs, (self.inputs, self.neurons))
        self.bias = np.random.normal(0, float(2)/self.inputs, (self.neurons))

    def activation(self, S, derivative=False):
        if self.activation_method == 'Sigmoid':
            if derivative:
                return S * (1 - S)
            else:
                return 1 / (1 + np.exp(-S))
        # TODO: Linear layer

    def output(self, values):
        S = np.dot(values, self.weights)
        S = np.add(S, self.bias)
        return self.activation(S)

    def update_weights(self, delta, inputs):
        # Reshape inputs to form: u(i), transposed for input to each neuron
        inputs = inputs.reshape(len(inputs),1)

        # Update weights by w(i,j) = w(i,j) + LR * delta(j) * u(i)
        w_change = const.LEARNING_RATE * delta
        w_change = w_change * inputs
        self.weights += w_change

        # Update bias
        b_change = const.LEARNING_RATE * delta
        self.bias += b_change