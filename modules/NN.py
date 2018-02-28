import numpy as np
import modules.const as const

np.random.seed(const.RANDOM_SEED)

# NN Model - Neuron Layer
class Layer:
    def __init__(self, num_inputs, num_neurons, activation_method):
        self.inputs = num_inputs
        self.neurons = num_neurons
        self.activation_method = activation_method

        # Stores last adjustments for momentum
        self.last_change_w = np.zeros((self.inputs, self.neurons))
        self.last_change_b = np.zeros((self.neurons))

        # randomly assign neuron starting weights and bias (-2/inputs -> 2/inputs)
        self.weights = np.random.normal(0, float(2)/self.inputs, (self.inputs, self.neurons))
        self.bias = np.random.normal(0, float(2)/self.inputs, (self.neurons))

    def activation(self, S, derivative=False):
        # Sigmoid
        if self.activation_method == 'Sigmoid':
            if derivative:
                return S * (1 - S)
            else:
                return 1 / (1 + np.exp(-S))
        # Linear
        elif self.activation_method == 'Linear':
            if derivative:
                return 1
            else:
                return S

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

        # Momentum
        if const.MOMENTUM:
            self.weights += self.last_change_w * const.MOMENTUM_ALPHA
            self.bias += self.last_change_b * const.MOMENTUM_ALPHA
            # Record changes for next iteration
            self.last_change_w = w_change
            self.last_change_b = b_change
