import numpy as np
import math

def get_labeled_objects(np_objects, np_labels):
    return np.c_[np_objects, np_labels]

def get_object(labeled_object):
    return labeled_object[:-1]

def get_objects(labeled_objects):
    return labeled_objects[:,-1]

def init_weights(input_size, output_size):
    a = 2.0 / (input_size + output_size)
    w = np.random.uniform(-a, a, (input_size, output_size))
    return w

def get_label(labeled_object):
    return int(labeled_object[-1])

def init_output(output_layer_size, batch_size, batch_labeled_objects):
    output = np.zeros((output_layer_size, batch_size), dtype=int)
    for i in range(len(batch_labeled_objects)):
        output[i][get_label(batch_labeled_objects[i])] = 1
    return output

def soft_max(val, exp_summ):
    return math.exp(val)/exp_summ

def sigmoid(val):
    return 1 / (1 + np.exp(-val))

def normalize_images(images):
    new_images = list()
    for image in images:
        new_images.append(normalize_image(image))
    return new_images

def normalize_image(image):
    return [i / 255 for i in image]

