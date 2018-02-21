import numpy as np
import pandas as pd
import modules.const as const
import modules.data

def main():
    # Read in cleaned data CSV
    train_x, train_y, test_x, test_y = modules.data.read_data()

    # print "X_Train\n", train_x
    # print "Y_Train\n", train_y

    layer_input = NNLayer(const.NEURONS[0], const.NEURONS[0])
    layer_hidden = NNLayer(const.NEURONS[0], const.NEURONS[1])
    layer_output = NNLayer(const.NEURONS[1], const.NEURONS[2])

    print "Hidden weights"
    print layer_hidden.weights
    print "Output weights"
    print layer_output.weights

    # Train model
    for j in range(10):
        for idx, train_x_row in enumerate(train_x):
            expected = train_y[idx]

            # Feed forward
            vals = layer_hidden.output(train_x_row)
            prediction = layer_output.output(vals)
            # print "Fed Foward...\nInput:",train_x_row, "Expected:", train_y[idx], "Prediction:", prediction

            # Backward pass
            output_derivative = layer_output.activation(prediction, True)
            output_delta = (expected - prediction) * output_derivative    # output neuron delta

            hidden_delta = []
            for i, val in enumerate(vals):
                hidden_derivative = layer_hidden.activation(val, True)
                hidden_delta.append(((output_delta * layer_output.weights[i]) * hidden_derivative)[0])

            layer_hidden.update_weights(np.array(hidden_delta), train_x_row)
            layer_output.update_weights(np.array(output_delta), vals)

    print "Hidden weights"
    print layer_hidden.weights
    print "Output weights"
    print layer_output.weights

    print "TEST"
    v = layer_hidden.output(test_x[0])
    p = layer_output.output(v)
    print "Input:",test_x[0], "Expected:", test_y[0], "Prediction:", p
    v = layer_hidden.output(test_x[1])
    p = layer_output.output(v)
    print "Input:",test_x[1], "Expected:", test_y[1], "Prediction:", p




# Define NN Model
class NNLayer:
    def __init__(self, num_inputs, num_neurons):
        self.inputs = num_inputs
        self.neurons = num_neurons
        # randomly assign neuron starting weights and bias (-2/inputs -> 2/inputs)
        self.weights = np.random.normal(0, float(2)/self.inputs, (self.inputs, self.neurons))
        self.bias = np.random.normal(0, float(2)/self.inputs, (self.neurons))

    def activation(self, S, derivative=False):
        if const.ACTIVATION == 'Sigmoid':
            if derivative:
                return S * (1 - S)
            else:
                return 1 / (1 + np.exp(-S))

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
        self.weights = np.add(self.weights, w_change)

        # Update bias
        b_change = const.LEARNING_RATE * delta
        self.bias = np.add(self.bias, b_change)


if __name__ == '__main__':
    main()
