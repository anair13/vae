# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Creates variants of the LeNet model. For each function:

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:

        logits = lenet.lenet(images, is_training=False)
        probabilities = tf.nn.softmax(logits)
        predictions = tf.argmax(logits, 1)

  Args:
    images: A batch of `Tensors` of size [batch_size, height, width, channels].
    num_classes: the number of classes in the dataset.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
"""

from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def lenet(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):
  with tf.variable_scope(scope, reuse=None):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    print(net.get_shape())
    net = slim.flatten(net)

    net = slim.fully_connected(net, 1024, scope='fc3')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')
  return logits

def lenet3(images, shared, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):
  with tf.variable_scope(scope, reuse=None):
    # net = slim.conv2d(images, 32, [3, 3], scope='conv1a')
    # if shared:
    #   net = slim.conv2d(net, 32, [3, 3], scope='conv1a', reuse=True)
    # else:
    #   net = slim.conv2d(net, 32, [3, 3], scope='conv1b')
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 32, [3, 3], scope='conv2a')
    if shared:
      net = slim.conv2d(net, 32, [3, 3], scope='conv2a', reuse=True)
    else:
      net = slim.conv2d(net, 32, [3, 3], scope='conv2b')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    print(net.get_shape())
    net = slim.flatten(net)

    net = slim.fully_connected(net, 1024, scope='fc3')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')

  return logits


def fc(images, shared, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):
  with tf.variable_scope(scope, reuse=None):
    net = slim.flatten(images)
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')
  return logits
