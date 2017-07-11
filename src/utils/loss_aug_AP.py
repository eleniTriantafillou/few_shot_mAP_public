from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class LossAugmentedInferenceAP(object):
  """
  Loss augmented inference algorithm of Song et al.
  for the task loss of Average Precision (AP).

  """

  def __init__(self, phi_pos, phi_neg, epsilon, positive_update=True):
    """
    :param phi_pos: cosine similarities between the query and each positive point
    :param phi_neg: cosine similarities between the query and each negative point
    :param epsilon: float used by DLM (see the paper for details)
    :param positive_update: whether or not to perform positive update of DLM
    """

    num_pos = phi_pos.shape[0]
    num_neg = phi_neg.shape[0]
    self._num_pos = num_pos
    self._num_neg = num_neg

    B, G = self.compute_B_and_G(phi_pos, phi_neg)
    self._B = B
    self._G = G

    if positive_update:
      self._negative_update = -1
    else:
      self._negative_update = 1
    self._epsilon = epsilon

    H, d = self.compute_H_and_d()
    self._direction = d
    self._H = H

    ranking = self.recover_ranking(d)
    self._ranking = ranking

  @property
  def num_pos(self):
    return self._num_pos

  @property
  def num_neg(self):
    return self._num_neg

  @property
  def B(self):
    return self._B

  @property
  def G(self):
    return self._G

  @property
  def negative_update(self):
    return self._negative_update

  @property
  def epsilon(self):
    return self._epsilon

  @property
  def direction(self):
    return self._direction

  @property
  def H(self):
    return self._H

  @property
  def ranking(self):
    return self._ranking

  def compute_B_and_G(self, phi_pos, phi_neg):
    B = np.zeros((self.num_pos + 1, self.num_neg + 1))
    G = np.zeros((self.num_pos + 1, self.num_neg + 1))

    for i in range(1, self.num_pos + 1):
      for j in range(1, self.num_neg + 1):
        B[i, j] = B[i, j - 1] - (phi_pos[i - 1] - phi_neg[j - 1]) / float(
            self.num_pos * self.num_neg)
        G[i, j] = G[i - 1, j] + (phi_pos[i - 1] - phi_neg[j - 1]) / float(
            self.num_pos * self.num_neg)

    return B, G

  def compute_H_and_d(self):
    H = np.zeros((self.num_pos + 1, self.num_neg + 1))
    direction = np.zeros((self.num_pos + 1, self.num_neg + 1))
    for i in range(self.num_pos + 1):
      for j in range(self.num_neg + 1):
        if i == 0 and j == 0:
          H[i, j] = 0
          direction[i, j] = 0
          continue
        if i == 1 and j == 0:
          H[i, j] = self.epsilon * self.negative_update / float(self.num_pos)
          direction[i, j] = 1
          continue
        if i == 0 and j == 1:
          H[i, j] = 0
          direction[i, j] = -1
          continue
        if i == 0:  # but j > 1
          H[i, j] = H[i, j - 1] + self.G[i, j]
          direction[i, j] = -1
          continue

        _add_pos = self.epsilon * 1.0 / self.num_pos * i / float(
            i + j) * self.negative_update + self.B[i, j]
        if j == 0:
          H[i, j] = H[i - 1, j] + _add_pos
          direction[i, j] = 1
          continue
        if (H[i, j - 1] + self.G[i, j]) > (H[i - 1, j] + _add_pos):
          H[i, j] = H[i, j - 1] + self.G[i, j]
          direction[i, j] = -1
        else:
          H[i, j] = H[i - 1, j] + _add_pos
          direction[i, j] = 1
    return H, direction

  def recover_ranking(self, d):
    ranking = np.zeros((self.num_pos + self.num_neg))
    i = self.num_pos
    j = self.num_neg
    while (i >= 0 and j >= 0 and not (i == 0 and j == 0)):
      if d[i, j] == 1:
        ranking[i - 1] = i + j - 1
        i -= 1
      else:
        ranking[j + self.num_pos - 1] = i + j - 1
        j -= 1
    return ranking
