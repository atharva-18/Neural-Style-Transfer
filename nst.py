import numpy as np
import time

from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import sys

import keras
from keras.applications import VGG19

from scipy.optimize import fmin_l_bfgs_b
from imageio import imwrite

import pydot
from IPython.display import display
from PIL import Image
from skimage.transform import resize

import keras.backend as K

"""Hyperparameters"""

content_weight = 0.025
style_weight = 1.0
total_variation_weight = 1.0

image_height = 512
image_width = 512

"""Generating image arrays from user images."""
arguments = sys.argv[1:]

content_image_path = arguments[0]
style_image_path = arguments[1]

content_image = Image.open(content_image_path)
content_image = content_image.resize((image_width, image_height))

style_image = Image.open(style_image_path)
style_image = style_image.resize((image_width, image_height))

"""Reshaping image arrays"""

content_array = np.asarray(content_image, dtype='float32')
style_array = np.asarray(style_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)
style_array = np.expand_dims(style_array, axis=0)
print("Content image dimensions -> ", content_array.shape)
print("Style image dimensions ->", style_array.shape)

"""Subtracting mean values of RGB and converting to BGR form."""

content_array[:, :, :, 0] -= 103.939
content_array[:, :, :, 1] -= 116.779
content_array[:, :, :, 2] -= 123.68

style_array[:, :, :, 0] -= 103.939
style_array[:, :, :, 1] -= 116.779
style_array[:, :, :, 2] -= 123.68

#RGB to BGR
content_array = content_array[:, :, :, ::-1]
style_array = style_array[:, :, :, ::-1]

"""Defining variables and placeholders to be used."""

content_image = K.variable(content_array)
style_image = K.variable(style_array)
combination_image = K.placeholder((1, image_height, image_width, 3))

"""Concatenating everything into a single tensor."""

input_tensor = K.concatenate([content_image,
                              style_image,
                              combination_image], axis=0)

"""Using VGG19 as our pretrained model."""

model = VGG19(input_tensor=input_tensor, 
              weights='imagenet', include_top=False)

layers = dict([(layer.name, layer.output) for layer in model.layers])

"""Defining our loss variable."""

loss = K.variable(0.0)

"""A function that generates gram matrix of 'x'"""

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

"""Defining our loss functions."""

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = image_height * image_width
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = K.square(x[:, :image_height-1, :image_width-1, :] - 
                 x[:, 1:, :image_width-1, :])
    b = K.square(x[:, :image_height-1, :image_width-1, :] - x[:, :image_height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

"""Using features of block2_conv2 layer."""

layer_features = layers['block2_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss = loss + content_weight * content_loss(base_image_features,
                                      combination_features)

"""Feature layers to be used."""

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, : ,: ,:]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

"""Defining our gradients tensor."""

grads = K.gradients(loss, combination_image)

outputs = [loss]
outputs += grads
f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, image_height, image_width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

"""Defining our evaluator class."""

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

"""Creating an instance of Evaluator."""

evaluator = Evaluator()

"""Generating a random white noise image."""

x = np.random.uniform(0, 255, (1, image_height, image_width, 3)) - 128.

"""Reducing loss and generating final image over iterations."""

iterations = 10

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun = 20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

"""Deprocessing the image."""

x = x.reshape((image_height, image_width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

Image.fromarray(x)
imwrite('result.png', x)
print("Image saved as result.png")
