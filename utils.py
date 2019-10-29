import numpy as np

def shuffle_samples(a, b):
    random_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(random_state)
    np.random.shuffle(b)
    return a, b

def init_weights(input_size, output_size):
    a = 2.0 / (input_size + output_size)
    w = np.random.uniform(-a, a, (input_size, output_size))
    return w

def relu(X):
    return X * (X > 0)

def relu_der(X):

    return 1. * (X > 0)

def sigmoid(val):
    return 1 / (1 + np.exp(-val))

def sigmoid_der(val):
    return val * (1 - val)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def normalize_images(images):
    new_images = list()
    for image in images:
        new_images.append(normalize_image(image))
    return new_images

def normalize_image(image):
    return [i / 255 for i in image]

