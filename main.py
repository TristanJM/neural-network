import numpy as np
import pandas as pd
import copy
import modules.const as const
import modules.data
import modules.NN as NN

np.random.seed(const.RANDOM_SEED)

def main():
    # Read in cleaned data CSV
    train_x, train_y, val_x, val_y, test_x, test_y = modules.data.read_data()

    # Generate NN with random starting weights
    layer_hidden = NN.Layer(const.NEURONS[0], const.NEURONS[1], 'Sigmoid')
    layer_output = NN.Layer(const.NEURONS[1], const.NEURONS[2], 'Linear')

    # Print model weights
    def show_weights(info_str):
        print ">> Weights after %s:" % (info_str)
        print "Hidden weights:\n{}\nOutput weights:\n{}".format(layer_hidden.weights, layer_output.weights)

    load_model(layer_hidden, layer_output)
    # show_weights('load')

    # Train model
    j = 0
    if const.TRAIN:
        overfit = False
        last_val_err = None
        while j < const.MAX_EPOCHS and not overfit:

            # Bold driver - automatic Learning Rate adjustment
            if const.BOLD_DRIVER:
                adjustment_needed = True
                adjustment_count = 0
                while adjustment_needed and adjustment_count < 10:
                    layer_hidden_BD = copy.deepcopy(layer_hidden)
                    layer_output_BD = copy.deepcopy(layer_output)

                    err_before = eval_model(j, train_x, train_y, layer_hidden_BD, layer_output_BD)

                    # Train on train data
                    for idx, train_x_row in enumerate(train_x):
                        # Backpropagate - feed forward/back and calculate weights
                        vals, out_delta, hid_delta = backprop(train_x_row, train_y[idx], layer_hidden, layer_output)

                        # Update weights
                        layer_hidden_BD.update_weights(np.array(hid_delta), train_x_row)
                        layer_output_BD.update_weights(np.array(out_delta), vals)

                    err_after = eval_model(j, train_x, train_y, layer_hidden_BD, layer_output_BD)

                    msg = ""
                    if err_after > err_before:
                        msg += "Higher so LR decrease"
                    else:
                        msg += "Lower so LR increase"
                    # print "Error on epoch",j,"::", err_before, '->', err_after, msg
                    if err_after > err_before:
                        # If error goes up, reduce the Learning Rate
                        if const.LEARNING_RATE * const.BOLD_DRIVER_DECREASE != 0.0:
                            const.LEARNING_RATE *= const.BOLD_DRIVER_DECREASE
                        adjustment_needed = True
                        adjustment_count += 1
                        # print "BOLD DRIVER REDUCED LR TO", const.LEARNING_RATE
                    else:
                        # If the error goes down, Learning Rate may be too small
                        adjustment_needed = False
                        const.LEARNING_RATE *= const.BOLD_DRIVER_INCREASE
                        # print "BOLD DRIVER INCREASED LR TO", const.LEARNING_RATE
                        clone_layer(layer_hidden_BD, layer_hidden)
                        clone_layer(layer_output_BD, layer_output)
            else:
                # Train on train data
                for idx, train_x_row in enumerate(train_x):
                    # Backpropagate - feed forward/back and calculate weights
                    vals, out_delta, hid_delta = backprop(train_x_row, train_y[idx], layer_hidden, layer_output)

                    # Update weights
                    layer_hidden.update_weights(np.array(hid_delta), train_x_row)
                    layer_output.update_weights(np.array(out_delta), vals)

            # print "LR", const.LEARNING_RATE

            if j % const.VALIDATION_EPOCHS == 0:
                err = eval_model(j, val_x, val_y, layer_hidden, layer_output, 'validation')
                if last_val_err is not None and err >= last_val_err:
                    overfit = True
                last_val_err = err
            j += 1

        # show_weights('training')
        save_model(layer_hidden, layer_output)

    error = eval_model(const.MAX_EPOCHS, test_x, test_y, layer_hidden, layer_output, 'test')
    prediction = predict(test_x, layer_hidden, layer_output)

    # Plot on graph
    modules.data.plot(prediction, test_y, 'scatter')

def backprop(train_x_row, expected, layer_hidden, layer_output):
    # Feed forward
    vals = layer_hidden.output(train_x_row)
    prediction = layer_output.output(vals)

    # Backward pass - output neuron
    output_derivative = layer_output.activation(prediction, True)
    output_delta = (expected - prediction) * output_derivative    # output neuron delta

    # Backward pass - hidden neurons
    hidden_delta = []
    for i, val in enumerate(vals):
        hidden_derivative = layer_hidden.activation(val, True)
        hidden_delta.append(((output_delta * layer_output.weights[i]) * hidden_derivative)[0])

    return [vals, output_delta, hidden_delta]

# Calculate model error statistic
def eval_model(epoch_num, x_data, y_data, layer_hid, layer_out, eval_type=''):
    p = predict(x_data, layer_hid, layer_out)

    sq_err = (y_data - p)**2
    mse = np.mean(sq_err)

    if len(eval_type) > 0:
        print 'Epoch %04d: %.5f MSE (%.5f RMSE) on %s set' % (epoch_num, mse, np.sqrt(mse), eval_type)
    return mse

# Use model to predict on given X data
def predict(data, layer_hid, layer_out):
    prediction = []
    for row in data:
        vals = layer_hid.output(row)
        prediction.append(layer_out.output(vals))
    return np.array(prediction)

# Save model weights to text file
def save_model(layer_hidden, layer_output):
    if len(const.MODEL_NAME) > 0:
        np.savetxt(const.MODEL_DIR + 'model_' + const.MODEL_NAME + '_hidden.txt', layer_hidden.weights)
        np.savetxt(const.MODEL_DIR + 'model_' + const.MODEL_NAME + '_output.txt', layer_output.weights)
        np.savetxt(const.MODEL_DIR + 'model_' + const.MODEL_NAME + '_hidden_bias.txt', layer_hidden.bias)
        np.savetxt(const.MODEL_DIR + 'model_' + const.MODEL_NAME + '_output_bias.txt', layer_output.bias)

# Load model from previous train
def load_model(layer_hidden, layer_output):
    if len(const.LOAD_MODEL) > 0:
        try:
            print "Loading model..."
            # Weights
            loaded_hidden = np.loadtxt(const.MODEL_DIR + 'model_' + const.LOAD_MODEL + '_hidden.txt')
            loaded_output = np.loadtxt(const.MODEL_DIR + 'model_' + const.LOAD_MODEL + '_output.txt')
            loaded_output = loaded_output.reshape(len(loaded_output),1)

            layer_hidden.weights = loaded_hidden
            layer_output.weights = loaded_output

            # Biases
            hidden_bias = np.loadtxt(const.MODEL_DIR + 'model_' + const.LOAD_MODEL + '_hidden_bias.txt')
            output_bias = np.loadtxt(const.MODEL_DIR + 'model_' + const.LOAD_MODEL + '_output_bias.txt')
            output_bias = output_bias.reshape(1)

            layer_hidden.bias = hidden_bias
            layer_output.bias = output_bias
        except IOError as e:
            print e

def clone_layer(layer1, layer2):
    layer2.weights = layer1.weights
    layer2.bias = layer1.bias
    layer2.last_change_w = layer1.last_change_w
    layer2.last_change_b = layer1.last_change_b

if __name__ == '__main__':
    main()
