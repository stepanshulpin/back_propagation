import numpy as np
from utils import sigmoid, init_weights, shuffle_samples, softmax, sigmoid_der


class Network(object):

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.weight_first = init_weights(hidden_layer_size, input_layer_size)
        self.weight_second = init_weights(output_layer_size, hidden_layer_size)

        self.learning_rate = 1
        self.batch_size = 1

    # required processing
    def train(self, objects, labels, epochs, learning_rate, batch_size):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        err = 0

        np_objects = np.array(objects)
        np_labels = np.array(labels)

        N = len(np_objects)

        for x in range(epochs):
            success = 0
            np_objects, np_labels = shuffle_samples(np_objects, np_labels)
            for n in range(0, N, batch_size):
                cur_batch_size = min(batch_size, N - n)
                batch_labeled_objects = np_objects[n: n + cur_batch_size]
                batch_labeled_labels = np_labels[n: n + cur_batch_size]
                outputs = np.zeros((cur_batch_size, self.output_layer_size))
                hidden_neuron_values = (sigmoid(np.dot(self.weight_first, batch_labeled_objects.T))).T
                output_neuron_values = (softmax(np.dot(self.weight_second, hidden_neuron_values.T))).T
                for i in range(cur_batch_size):
                    output = np.zeros(self.output_layer_size, dtype=int)
                    output[batch_labeled_labels[i]] = 1
                    outputs[i] = output
                    if batch_labeled_labels[i] == np.argmax(output_neuron_values[i]):
                        success += 1

                self.back_propagation_batch(batch_labeled_objects, outputs, hidden_neuron_values, output_neuron_values,
                                            cur_batch_size)
            print("Epoch " + str(x))

            err = success / N

            print("Error " + str(1-err))
        print("Train accuracy " + str(err))
        print("Train error " + str(1-err))

    def test(self, objects, labels):
        success = 0
        np_objects = np.array(objects)
        np_labels = np.array(labels)
        for i in range(len(np_objects)):
            obj = np_objects[i]
            hidden_neuron_values = self.compute_hidden_layer(obj)
            output_neuron_values = self.compute_output_layer(hidden_neuron_values)
            if np_labels[i] == np.argmax(output_neuron_values):
                success += 1
        err = success / len(np_objects)
        print("Test accuracy " + str(err))
        print("Test error " + str(1 - err))

    def compute_hidden_layer_batch(self, obj):
        return sigmoid(np.dot(obj, self.weight_first.transpose()))

    def compute_output_layer_batch(self, hidden_neuron_values):
        output_neuron_values = np.exp(np.dot(hidden_neuron_values, self.weight_second.transpose()))
        return output_neuron_values / np.sum(output_neuron_values)

    def compute_hidden_layer(self, obj):
        return sigmoid(np.dot(obj, self.weight_first.transpose()))

    def compute_output_layer(self, hidden_neuron_values):
        output_neuron_values = np.exp(np.dot(hidden_neuron_values, self.weight_second.transpose()))
        return output_neuron_values / np.sum(output_neuron_values)

    def back_propagation_batch(self, objects, outputs, hidden_neuron_values, output_neuron_values, N):
        output_neuron_deltas = outputs - output_neuron_values
        self.correct_second_layer_weights_batch(output_neuron_deltas, hidden_neuron_values, N)
        self.correct_first_layer_weights_batch(objects, output_neuron_deltas, hidden_neuron_values, N)

    def correct_second_layer_weights_batch(self, output_neuron_deltas, hidden_neuron_values, N):
        grad = np.dot(output_neuron_deltas.T, hidden_neuron_values) / N
        self.weight_second = self.weight_second + self.learning_rate * grad

    def correct_first_layer_weights_batch(self, objects, output_neuron_deltas, hidden_neuron_values, N):
        delta = np.dot(output_neuron_deltas, self.weight_second) * sigmoid_der(hidden_neuron_values)
        grad = np.dot(delta.T, objects) / N
        self.weight_first = self.weight_first + self.learning_rate * grad
