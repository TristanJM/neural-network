import numpy as np
import pandas as pd
import modules.const as const
import modules.data
import modules.NN as NN

np.random.seed(const.RANDOM_SEED)

def main():
    # Read in cleaned data CSV
    train_x, train_y, test_x, test_y = modules.data.read_data()

    # print "X_Train\n", train_x
    # print "Y_Train\n", train_y
    # print "X_Test:\n", test_x
    # print "Y_Test:\n", test_y

    layer_hidden = NN.Layer(const.NEURONS[0], const.NEURONS[1], 'Sigmoid')
    layer_output = NN.Layer(const.NEURONS[1], const.NEURONS[2], 'Sigmoid')

    # Print model weights
    def show_weights():
        print "Hidden weights:\n{}\nOutput weights:\n{}".format(layer_hidden.weights, layer_output.weights)

    # Calculate model error statistic
    def eval_model(test_x, test_y):
        p = predict(test_x, layer_hidden, layer_output)

        sq_err = (test_y - p)**2
        mse = np.mean(sq_err)
        print 'Epoch %04d: %.5f MSE (%.5f RMSE)' % (j, mse, np.sqrt(mse))
        return mse

    # Train model
    for j in range(200):
        for idx, train_x_row in enumerate(train_x):
            expected = train_y[idx]

            # Feed forward
            vals = layer_hidden.output(train_x_row)
            prediction = layer_output.output(vals)

            # Backward pass
            output_derivative = layer_output.activation(prediction, True)
            output_delta = (expected - prediction) * output_derivative    # output neuron delta

            hidden_delta = []
            for i, val in enumerate(vals):
                hidden_derivative = layer_hidden.activation(val, True)
                hidden_delta.append(((output_delta * layer_output.weights[i]) * hidden_derivative)[0])

            layer_hidden.update_weights(np.array(hidden_delta), train_x_row)
            layer_output.update_weights(np.array(output_delta), vals)

        if j % 50 == 0:
            eval_model(test_x, test_y)

    error = eval_model(test_x, test_y)
    prediction = predict(test_x, layer_hidden, layer_output)

    # Plot on graph
    modules.data.plot(prediction, test_y)

# Use model to predict on given X data
def predict(data, layer_hidden, layer_output):
    prediction = []
    for row in data:
        vals = layer_hidden.output(row)
        prediction.append(layer_output.output(vals))
    return np.array(prediction)


if __name__ == '__main__':
    main()
