import numpy as np

from utils import get_label, get_labeled_objects, get_object, sigmoid, init_weights


class Network(object):

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.weight_first = init_weights(hidden_layer_size, input_layer_size)
        self.weight_second = init_weights(output_layer_size, hidden_layer_size)

        self.hidden_neuron_values = np.empty(hidden_layer_size)
        self.output_neuron_values = np.empty(output_layer_size)

        self.hidden_neuron_deltas = np.empty(hidden_layer_size)
        self.output_neuron_deltas = np.empty(output_layer_size)

        self.learning_rate = 1
        self.batch_size = 1

    # required processing
    def train(self, objects, labels, epochs, learning_rate, batch_size):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        err = 0

        np_objects = np.array(objects)
        np_labels = np.array(labels)

        for x in range(epochs):
            success = 0

            labeled_objects = get_labeled_objects(np_objects, np_labels)
            N = len(labeled_objects)
            np.random.shuffle(labeled_objects)

            for labeled_object in labeled_objects:

                output = np.zeros(self.output_layer_size, dtype=int)
                output[get_label(labeled_object)] = 1

                obj = get_object(labeled_object)
                self.compute_hidden_layer(obj)
                self.compute_output_layer()
                if get_label(labeled_object) == np.argmax(self.output_neuron_values):
                    success += 1
                self.back_propagation(obj, output)

            print("Epoch " + str(x))

            err = 1 - success / N

            print("Error " + str(err))
        print("Train error " + str(err))

    def test(self, objects, labels):
        success = 0
        np_objects = np.array(objects)
        np_labels = np.array(labels)
        labeled_objects = get_labeled_objects(np_objects, np_labels)
        for labeled_object in labeled_objects:
            obj = get_object(labeled_object)
            self.compute_hidden_layer(obj)
            self.compute_output_layer()
            if get_label(labeled_object) == np.argmax(self.output_neuron_values):
                success += 1
        err = success / len(labeled_objects)
        print("Test error " + str(1 - err))

    def compute_hidden_layer(self, obj):
        self.hidden_neuron_values = sigmoid(np.dot(obj, self.weight_first.transpose()))

    def compute_output_layer(self):
        self.output_neuron_values = np.exp(np.dot(self.hidden_neuron_values, self.weight_second.transpose()))
        self.output_neuron_values = self.output_neuron_values / np.sum(self.output_neuron_values)

    def back_propagation(self, obj, output):
        self.output_neuron_deltas = output - self.output_neuron_values
        self.correct_second_layer_weights()
        self.hidden_neuron_deltas = self.hidden_neuron_values * (1 - self.hidden_neuron_values) * np.dot(
            self.output_neuron_deltas, self.weight_second)
        self.correct_first_layer_weights(obj)

    def correct_first_layer_weights(self, obj):
        self.weight_first = self.weight_first + self.learning_rate * np.dot(self.hidden_neuron_deltas.reshape(
            self.hidden_layer_size, 1), obj.reshape(1, self.input_layer_size))

    def correct_second_layer_weights(self):
        self.weight_second = self.weight_second + self.learning_rate * np.dot(self.output_neuron_deltas.reshape(
            self.output_layer_size, 1), self.hidden_neuron_values.reshape(1, self.hidden_layer_size))
