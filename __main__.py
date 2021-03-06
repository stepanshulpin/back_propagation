from KeranNetwork import KerasNetwork
from MNISTLoader import MNISTLoader
from Network import Network

import sys
import argparse


def create_parser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-hidden_size', default=300)
    parser.add_argument ('-epochs', default=15)
    parser.add_argument ('-rate', default=0.1)
    parser.add_argument ('-batch', default=128)

    return parser

parser = create_parser()
namespace = parser.parse_args(sys.argv[1:])

ml = MNISTLoader('D:\samples')

print('My Network')
network = Network(ml.img_size, int(namespace.hidden_size), ml.outputs)
network.train(ml.images_train_norm, ml.labels_train, int(namespace.epochs), float(namespace.rate), int(namespace.batch))
network.test(ml.images_test_norm, ml.labels_test)

print('Keras Network')
keras_network = KerasNetwork(ml.img_size, int(namespace.hidden_size), ml.outputs)
keras_network.train(ml.images_train_norm, ml.labels_train, int(namespace.epochs), float(namespace.rate), int(namespace.batch))
keras_network.test(ml.images_test_norm, ml.labels_test)