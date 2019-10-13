from MNISTLoader import MNISTLoader
from Network import Network

ml = MNISTLoader('D:\samples')

network = Network(ml.img_size, 16, ml.outputs)

network.train(ml.images_train_norm, ml.labels_train, 10, 0.1, 1)

network.test(ml.images_test_norm, ml.labels_test)