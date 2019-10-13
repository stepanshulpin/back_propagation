from mnist import MNIST

from utils import normalize_images


class MNISTLoader:

    def __init__(self, path):
        mndata = MNIST(path)
        images_train, self.labels_train = mndata.load_training()
        images_test, self.labels_test = mndata.load_testing()

        self.images_train_norm = normalize_images(images_train)
        self.images_test_norm = normalize_images(images_test)

        self.img_size = len(images_train[0])
        self.outputs = 10

    

