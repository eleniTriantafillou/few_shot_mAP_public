from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
import tensorflow.contrib.slim as slim

from src.models.model import Model


class SiameseModel(Model):
  """
  Siamese model with cross-entropy loss function.
  """

  def __init__(self, config, reuse=False):
    self._first_inds = tf.placeholder(tf.int32, [None])
    self._second_inds = tf.placeholder(tf.int32, [None])
    y_dim = 1
    if config.loss_function == "cross_entropy":
      y_dim = 2
    self._y = tf.placeholder(tf.float32, [None, y_dim])
    super(SiameseModel, self).__init__(config, reuse)
    self._siamese_accuracy = self.compute_siamese_accuracy()

  def join_branches(self, feats_A, feats_B):
    feats_A = tf.truediv(
        feats_A, tf.sqrt(tf.reduce_sum(tf.square(feats_A), 1, keep_dims=True)))
    feats_B = tf.truediv(
        feats_B, tf.sqrt(tf.reduce_sum(tf.square(feats_B), 1, keep_dims=True)))
    if self.config.join_branches == "concat":
      pair_feats = tf.concat(1, [feats_A, feats_B])
    elif self.config.join_branches == "abs_diff":
      pair_feats = tf.abs(feats_A - feats_B)
    return pair_feats

  def get_siamese_prediction(self):
    feats_A = tf.gather(self.feats, self.first_inds)
    feats_B = tf.gather(self.feats, self.second_inds)
    if self.config.loss_function == "cross_entropy":
      joined_feats = self.join_branches(feats_A, feats_B)
      pred = slim.fully_connected(joined_feats, 2, activation_fn=None)
    return pred

  def compute_loss(self):
    self._siamese_prediction = self.get_siamese_prediction()
    if self.config.loss_function == "cross_entropy":
      loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits=self.siamese_prediction,
                                                  labels=self.y))
    return loss

  def compute_siamese_accuracy(self):
    pred_softmax = tf.nn.softmax(self.siamese_prediction)
    correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(self.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

  @property
  def first_inds(self):
    return self._first_inds

  @property
  def second_inds(self):
    return self._second_inds

  @property
  def y(self):
    return self._y

  @property
  def siamese_accuracy(self):
    return self._siamese_accuracy

  @property
  def siamese_prediction(self):
    return self._siamese_prediction
