from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from src.models.model import Model
from src.utils.loss_aug_AP import LossAugmentedInferenceAP


class MeanAveragePrecisionModel(Model):
  """
  Our model for optimizing Mean Average Precision.
  """

  def __init__(self, config, reuse=False):
    super(MeanAveragePrecisionModel, self).__init__(config, reuse)

  def compute_loss(self):
    if self.config.optimization_framework == "SSVM":
      loss = self.mAP_score_aug * self.config.alpha - self.mAP_score_GT
    elif self.config.optimization_framework == "DLM":  # Direct Loss minimization
      loss = (1 / self.config.epsilon) * (
          self.mAP_score_aug * self.config.alpha - self.mAP_score_std)
      if not self.config.positive_update:
        loss *= -1
    else:
      raise ValueError("Unknown optimization framework {}".format(
          self.config.optimization_framework))
    return loss

  def perform_loss_augmented_inference(self, sess, batch):
    batch_size = len(batch["labels"])
    num_pos, num_neg, pos_inds, neg_inds = self.get_positive_negative_splits(
        batch)
    _feed_dict = {
        self.x: batch["imgs"],
        self.n_queries_to_parse: self.config.batch_size,
        self.num_pos: num_pos,
        self.num_neg: num_neg,
        self.pos_inds: pos_inds,
        self.neg_inds: neg_inds
    }
    phi_pos, phi_neg, skipped_queries = sess.run(
        [self.phi_pos, self.phi_neg, self.skipped_queries],
        feed_dict=_feed_dict)
    Y_aug = np.zeros((batch_size, np.max(num_pos), np.max(num_neg)))
    for qq in range(batch_size):
      if skipped_queries[qq] == 1:
        print("Skipped {}".format(qq))
        continue
      q_phi_pos = phi_pos[qq][:num_pos[qq]]
      q_phi_neg = phi_neg[qq][:num_neg[qq]]

      loss_aug_AP_algo = LossAugmentedInferenceAP(q_phi_pos, q_phi_neg,
                                                  self.config.epsilon,
                                                  self.config.positive_update)
      q_Y_aug = -1 * loss_aug_AP_algo.direction[1:, 1:]
      Y_aug[qq, :num_pos[qq], :num_neg[qq]] = q_Y_aug

    return Y_aug

  @property
  def config(self):
    return self._config
