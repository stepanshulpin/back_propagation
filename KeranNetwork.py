from datetime import datetime
from keras import optimizers, Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical


class KerasNetwork(object):

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.learning_rate = 1
        self.batch_size = 1
        self.model = Sequential()
        self.model.add(Dense(self.hidden_layer_size, activation='relu', input_shape=(self.input_layer_size,)))
        self.model.add(Dense(self.output_layer_size, activation='softmax'))

    def train(self, objects, labels, epochs, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        np_objects = np.array(objects)
        np_labels = to_categorical(np.array(labels))

        sgd = optimizers.SGD(lr=learning_rate, momentum=0.0, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

        time_start = datetime.now()
        self.model.fit(np_objects, np_labels, epochs=epochs, batch_size=batch_size)
        time = datetime.now() - time_start
        train_result = self.model.evaluate(np_objects, np_labels)

        print("Train time " + str(time))
        print("Train accuracy " + str(train_result))

    def test(self, objects, labels):
        np_objects = np.array(objects)
        np_labels = to_categorical(np.array(labels))
        test_result = self.model.evaluate(np_objects, np_labels)
        print("Test accuracy " + str(test_result))