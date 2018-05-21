"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py

The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim


def AlexNet_arg_scope(weight_decay=0.00004,
                      stddev=0.1,
                      batch_norm_var_collection='moving_vars',
                      is_training=True):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.max_pool2d], padding='VALID', kernel_size=[3, 3], stride=2) as sc:
            return sc

class AlexNet(object):
    """Implementation of the AlexNet."""
    def __init__(self, x, is_training):
        """Create the graph of the AlexNet model.
        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        # Call the create function to build the computational graph of AlexNet
        self.is_training = is_training
        self.create()

    def create(self):
        """Create the network graph."""
        with slim.arg_scope(AlexNet_arg_scope(is_training=self.is_training)):
            conv1 = slim.conv2d(self.X, 64, [11, 11], stride=4, padding='VALID', scope='conv1')
            pool1 = slim.max_pool2d(conv1, scope='pool1')

            conv2 = slim.conv2d(pool1, 256, [5, 5], stride=1, padding='SAME', scope='conv2')
            pool2 = slim.max_pool2d(conv2, scope='pool2')

            conv3 = slim.conv2d(pool2, 384, [3, 3], stride=1, padding='SAME', scope='conv3')
            conv4 = slim.conv2d(conv3, 384, [3, 3], stride=1, padding='SAME', scope='conv4')

            conv5 = slim.conv2d(conv4, 256, [3, 3], stride=1, padding='SAME', scope='conv5')

            pool5 = slim.max_pool2d(conv5, scope='pool5')
            flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])

            self.flattened = flattened
            fc6 = slim.fully_connected(flattened, 4096, scope='fc6')
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
            self.fc8 = slim.fully_connected(fc7, 4, activation_fn=None, scope='fc8')
