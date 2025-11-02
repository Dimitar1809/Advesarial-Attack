# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Model(object):
  """AlexNet model."""

  def __init__(self, mode):
    """AlexNet constructor.

    Args:
      mode: One of 'train' and 'eval'.
    """
    self.mode = mode
    self._build_model()

  def add_internal_summaries(self):
    pass

  def _build_model(self):
    assert self.mode == 'train' or self.mode == 'eval'
    """Build the core model within the graph."""
    with tf.variable_scope('input'):

      self.x_input = tf.placeholder(
        tf.float32,
        shape=[None, 32, 32, 3])

      self.y_input = tf.placeholder(tf.int64, shape=None)

      # Per-image standardization (keep same preprocessing as original)
      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                               self.x_input)
      x = input_standardized

    # AlexNet architecture for CIFAR-10
    # Conv Layer 1: 32x32x3 -> 16x16x64
    with tf.variable_scope('conv1'):
      x = self._conv('conv', x, 3, 3, 64, [1, 2, 2, 1])
      x = self._relu(x, 0.0)
      if self.mode == 'train':
        x = tf.nn.dropout(x, keep_prob=0.9)

    # Conv Layer 2: 16x16x64 -> 8x8x192
    with tf.variable_scope('conv2'):
      x = self._conv('conv', x, 3, 64, 192, [1, 2, 2, 1])
      x = self._relu(x, 0.0)
      if self.mode == 'train':
        x = tf.nn.dropout(x, keep_prob=0.9)

    # Conv Layer 3: 8x8x192 -> 8x8x384
    with tf.variable_scope('conv3'):
      x = self._conv('conv', x, 3, 192, 384, [1, 1, 1, 1])
      x = self._relu(x, 0.0)
      if self.mode == 'train':
        x = tf.nn.dropout(x, keep_prob=0.9)

    # Conv Layer 4: 8x8x384 -> 8x8x256
    with tf.variable_scope('conv4'):
      x = self._conv('conv', x, 3, 384, 256, [1, 1, 1, 1])
      x = self._relu(x, 0.0)
      if self.mode == 'train':
        x = tf.nn.dropout(x, keep_prob=0.9)

    # Conv Layer 5: 8x8x256 -> 4x4x256
    with tf.variable_scope('conv5'):
      x = self._conv('conv', x, 3, 256, 256, [1, 2, 2, 1])
      x = self._relu(x, 0.0)
      if self.mode == 'train':
        x = tf.nn.dropout(x, keep_prob=0.9)

    # Flatten
    x = tf.reshape(x, [tf.shape(x)[0], -1])

    # Fully connected layer 1
    with tf.variable_scope('fc1'):
      x = self._fully_connected(x, 4096)
      x = self._relu(x, 0.0)
      if self.mode == 'train':
        x = tf.nn.dropout(x, keep_prob=0.5)

    # Fully connected layer 2
    with tf.variable_scope('fc2'):
      x = self._fully_connected(x, 4096)
      x = self._relu(x, 0.0)
      if self.mode == 'train':
        x = tf.nn.dropout(x, keep_prob=0.5)

    # Output layer
    with tf.variable_scope('logit'):
      self.pre_softmax = self._fully_connected(x, 10)

    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('costs'):
      self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)