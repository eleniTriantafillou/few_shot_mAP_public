from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import tensorflow as tf


class Evaluator(object):
  """
  Class for Evaluation.
  """

  def __init__(self, config, model, dataset, sess):
    self._config = config
    self._model = model
    self._dataset = dataset
    self._sess = sess

  @property
  def config(self):
    return self._config

  @property
  def model(self):
    return self._model

  @property
  def dataset(self):
    return self._dataset

  @property
  def sess(self):
    return self._sess

  def eval_mAP(self, batch):
    imgs = batch["imgs"]
    labels = batch["labels"]
    batch_size = len(labels)

    num_pos, num_neg, pos_inds, neg_inds = self.model.get_positive_negative_splits(
        batch)
    _feed_dict = {
        self.model.x: imgs,
        self.model.n_queries_to_parse: self.config.batch_size,
        self.model.num_pos: num_pos,
        self.model.num_neg: num_neg,
        self.model.pos_inds: pos_inds,
        self.model.neg_inds: neg_inds
    }
    eval_skipped_queries, eval_phi_pos, eval_phi_neg = self.sess.run(
        [self.model.skipped_queries, self.model.phi_pos, self.model.phi_neg],
        feed_dict=_feed_dict)
    skip = list(np.where(eval_skipped_queries == 1)[0])

    query_APs = []
    for q in range(batch_size):
      if q in skip:
        continue
      q_phi_pos = eval_phi_pos[q][:num_pos[q]]
      q_phi_neg = eval_phi_neg[q][:num_neg[q]]

      y_true = np.concatenate(
          (np.ones((q_phi_pos.shape[0])), np.zeros((q_phi_neg.shape[0]))),
          axis=0)
      y_scores = np.concatenate((q_phi_pos, q_phi_neg), axis=0)
      AP = self.apk(y_true, y_scores)
      query_APs.append(AP)

    mAP = None  # If skipped all queries
    if len(query_APs) > 0:
      mAP = sum(query_APs) / float(len(query_APs))
    return mAP

  def apk(self, relevant, scores, k=10000):
    """
    Computes the average precision at k between two lists of items.

    :param relevant: A list of truth values for each point
    :param scores: A list of predicted scores for each point
    :param k: (optional) The maximum number of predicted elements
    :return:
    """
    assert len(relevant) == len(scores)
    assert relevant.ndim == 1 and scores.ndim == 1
    ranks = np.argsort(scores)[::-1]  # for example [1, 0, 3, 2]
    # sort in descending order the parallel lists relevant and scores according to ranks
    actual = np.array(relevant)[ranks]
    predicted = np.array(scores)[ranks]

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
      if actual[i]:
        num_hits += 1.0
        score += num_hits / (i + 1.0)
    num_relevant = len(np.where(actual == 1)[0])
    if min(num_relevant, k) > 0:
      AP = score / min(num_relevant, k)
    else:
      AP = None
    return AP

  def compute_confidence_interval(self, data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

  def eval_fewshot_classif(self, K, N, num_samples=None):
    accuracy = []
    if num_samples is None:
      num_samples = self.config.num_fewshot_samples
    for r in range(num_samples):
      ref_paths, query_paths, labels = self.dataset.create_KshotNway_classification_episode(
          K, N)

      if K == 1:
        accuracy.append(
            self.run_1shotNway_classif(ref_paths, query_paths, labels))
      elif K > 1:
        accuracy.append(
            self.run_KshotNway_classif(ref_paths, query_paths, labels))
      else:
        raise ValueError("Expecting K >= 1.")
    accuracy = np.array(accuracy)
    mean, plus_minus = self.compute_confidence_interval(accuracy)
    print("{}-shot-{}-way classif accuracy: {} plus/minus {}%".format(
        K, N, mean, plus_minus))
    return mean, plus_minus

  def eval_oneshot_retrieval(self, N, n_per_class, num_samples=None):
    mAP = []
    if num_samples is None:
      num_samples = self.config.num_fewshot_samples
    for r in range(num_samples):
      paths, labels = self.dataset.create_oneshotNway_retrieval_episode(
          N, n_per_class)
      mAP.append(self.run_1shotNway_retrieval(paths, labels))
    mean, plus_minus = self.compute_confidence_interval(mAP)
    print("one-shot-{}-way retrieval mAP: {} plus/minus {}%".format(
        N, mean, plus_minus))
    return mean, plus_minus

  def run_1shotNway_classif(self, ref_paths, query_paths, labels):
    """
    Perform 1-shot N-way classification.

    :param ref_paths: A list of N lists, each containing the path of the single class references
    :param query_paths: A list of N lists, each containing the paths of the 19 class queries
    :param labels: A list of N labels, one for each class
    :return: mean_accuracy: Mean (across the queries) 1-shot N-way classification accuracy.
    """

    def add_to_images(imgs_array, new_img_array):
      if not len(new_img_array.shape) == 4:
        _h = new_img_array.shape[0]
        _w = new_img_array.shape[1]
        _c = new_img_array.shape[2]
        new_img_array = new_img_array.reshape((1, _h, _w, _c))
      if imgs_array.shape[0] == 0:
        imgs_array = new_img_array
      else:
        imgs_array = np.concatenate((imgs_array, new_img_array), axis=0)
      return imgs_array

    def read_data():
      ref_labels, query_labels = [], []
      K = len(ref_paths[0])  # number of representatives of each class
      for class_label in labels:
        ref_labels += [class_label] * K
        query_labels += [class_label] * (20 - K)

      imgs_queries, imgs_refs = np.array([]), np.array([])
      for class_rpaths, class_qpaths in zip(ref_paths, query_paths):
        for rpath in class_rpaths:
          img_array = self.dataset.load_img_as_array(rpath)
          imgs_refs = add_to_images(imgs_refs, img_array)
        for qpath in class_qpaths:
          img_array = self.dataset.load_img_as_array(qpath)
          imgs_queries = add_to_images(imgs_queries, img_array)
      return imgs_refs, imgs_queries, ref_labels, query_labels

    imgs_refs, imgs_queries, ref_labels, query_labels = read_data()

    num_queries = imgs_queries.shape[0]

    # computational graph for computing pairwise similarities
    feats_queries = tf.placeholder(tf.float32, [None, None])
    feats_references = tf.placeholder(tf.float32, [None, None])
    S = tf.matmul(
        feats_queries, feats_references,
        transpose_b=True)  # (num_queries, num_refs)

    feats_queries_ = self.sess.run(
        self.model.feats, feed_dict={self.model.x: imgs_queries})
    feats_references_ = self.sess.run(
        self.model.feats, feed_dict={self.model.x: imgs_refs})
    S_ = self.sess.run(
        S,
        feed_dict={
            feats_queries: feats_queries_,
            feats_references: feats_references_
        })
    pred_ind = np.argmax(S_, axis=1)
    correct = 0.0
    for i in range(num_queries):
      pred_label = ref_labels[pred_ind[i]]
      true_label = query_labels[i]
      if pred_label == true_label:
        correct += 1.0
    mean_accuracy = 100 * correct / float(num_queries)
    return mean_accuracy

  def run_KshotNway_classif(self, ref_paths, query_paths, labels):
    """
    Perform K-shot N-way classification.

    The algorithm for exploiting all K examples of each of the N new classes
    in order to make a classification decision is the following.
    Given a query point to be classified:
    For each class n \in N:
    Compute AP for the ranking of all K*N candidate points with the reference as query
    assuming the correct class is the nth
    (i.e. treat as relevant the points of the nth class and irrelevant everyhting else)
    At the end, classify the reference point into the class for which the average precision
    of the reference's ranking is highest when this class is treated as the groundtruth.

    :param ref_paths: A list of N lists, each containing the paths of the K references
    :param query_paths: A list of N lists, each containing the paths of the (20 - K) queries
    :param labels: A list of N labels, one for each class
    :return: mean_accuracy: Mean (across the queries) K-shot N-way classification accuracy.
    """

    N = len(labels)

    def add_to_images(imgs_array, new_img_array):
      if not len(new_img_array.shape) == 4:
        _h = new_img_array.shape[0]
        _w = new_img_array.shape[1]
        _c = new_img_array.shape[2]
        new_img_array = new_img_array.reshape((1, _h, _w, _c))
      if imgs_array.shape[0] == 0:
        imgs_array = new_img_array
      else:
        imgs_array = np.concatenate((imgs_array, new_img_array), axis=0)
      return imgs_array

    def read_data():
      ref_labels, query_labels = [], []
      K = len(ref_paths[0])  # number of representatives of each class
      for class_label in labels:
        ref_labels += [class_label] * K
        query_labels += [class_label] * (20 - K)

      imgs_queries, imgs_refs = np.array([]), np.array([])
      for class_rpaths, class_qpaths in zip(ref_paths, query_paths):
        for rpath in class_rpaths:
          img_array = self.dataset.load_img_as_array(rpath)
          imgs_refs = add_to_images(imgs_refs, img_array)
        for qpath in class_qpaths:
          img_array = self.dataset.load_img_as_array(qpath)
          imgs_queries = add_to_images(imgs_queries, img_array)
      return imgs_refs, imgs_queries, ref_labels, query_labels

    def get_splits(candidates_labels, query_class):
      """
      Separate the candidates into positive/negative for the
      query under the assumption that the query's class is
      query_class.

      :param candidates_labels: The labels of all K * N candidates
      :param query_class: The (assumed) class of the query
      :return: The splits for positive/negative set of the query
      under the assumption that it belongs to class query_class.
      """

      pos_inds, neg_inds = [], []
      for i in range(len(candidates_labels)):
        if candidates_labels[i] == query_class:
          pos_inds.append(i + 1)
        else:
          neg_inds.append(i + 1)
      num_pos, num_neg = np.array([len(pos_inds)]), np.array([len(neg_inds)])
      pos_inds = np.array(pos_inds).reshape((1, -1))
      neg_inds = np.array(neg_inds).reshape((1, -1))
      return num_pos, num_neg, pos_inds, neg_inds

    imgs_refs, imgs_queries, ref_labels, query_labels = read_data()
    num_queries = len(imgs_queries)

    predicted_classes = []  # becomes length num_queries
    for q in range(num_queries):
      query_img = imgs_queries[q].reshape(
          (1, self.config.height, self.config.width, self.config.channels))
      q_and_refs = np.concatenate((query_img, imgs_refs), axis=0)

      AP_for_different_ns = []  # list of length N eventually
      for n in range(N):
        num_pos, num_neg, pos_inds, neg_inds = get_splits(ref_labels, labels[n])
        _feed_dict = {
            self.model.x: q_and_refs,
            self.model.n_queries_to_parse:
            1,  # we are only interested in computing AP for query q's ranking
            self.model.num_pos: num_pos,
            self.model.num_neg: num_neg,
            self.model.pos_inds: pos_inds,
            self.model.neg_inds: neg_inds
        }
        phi_pos = self.sess.run(self.model.phi_pos, feed_dict=_feed_dict)
        phi_neg = self.sess.run(self.model.phi_neg, feed_dict=_feed_dict)
        q_phi_pos = phi_pos[0, :][:num_pos[0]]
        q_phi_neg = phi_neg[0, :][:num_neg[0]]
        y_true = np.concatenate(
            (np.ones((q_phi_pos.shape[0])), np.zeros((q_phi_neg.shape[0]))),
            axis=0)
        y_scores = np.concatenate((q_phi_pos, q_phi_neg), axis=0)
        y_scores = y_scores.reshape((y_scores.shape[0],))
        AP = self.apk(y_true, y_scores)
        AP_for_different_ns.append(AP)

      pred = np.argmax(AP_for_different_ns)
      predicted_classes.append(labels[pred])

    correct = 0.0
    for i in range(num_queries):
      pred_label = predicted_classes[i]
      true_label = query_labels[i]
      if pred_label == true_label:
        correct += 1.0
    mean_accuracy = 100 * correct / float(num_queries)
    return mean_accuracy

  def run_1shotNway_retrieval(self, paths, labels):
    """
    Perform 1-shot N-way retrieval.
    Given a "pool" of points, treat each one as a query and compute
    the AP of its ranking of the remaining points based on its
    predicted relevance to them. Report mAP across all queries.

    :param paths: A list of length 10 * N of paths of each point in the "pool"
    :param labels: A list of the corresponding labels for each point in the "pool"
    :return: mAP: The mean Average Precision over all queries in the "pool"
    """

    def add_to_images(imgs_array, new_img_array):
      if not len(new_img_array.shape) == 4:
        _h = new_img_array.shape[0]
        _w = new_img_array.shape[1]
        _c = new_img_array.shape[2]
        new_img_array = new_img_array.reshape((1, _h, _w, _c))
      if imgs_array.shape[0] == 0:
        imgs_array = new_img_array
      else:
        imgs_array = np.concatenate((imgs_array, new_img_array), axis=0)
      return imgs_array

    n_queries = len(paths)
    imgs = np.array([])
    for path in paths:
      img_array = self.dataset.load_img_as_array(path)
      imgs = add_to_images(imgs, img_array)

    batch = {}
    batch["labels"] = labels
    num_pos, num_neg, pos_inds, neg_inds = self.model.get_positive_negative_splits(
        batch)
    _feed_dict = {
        self.model.x: imgs,
        self.model.n_queries_to_parse: len(imgs),
        self.model.num_pos: num_pos,
        self.model.num_neg: num_neg,
        self.model.pos_inds: pos_inds,
        self.model.neg_inds: neg_inds
    }
    phi_pos = self.sess.run(self.model.phi_pos, feed_dict=_feed_dict)
    phi_neg = self.sess.run(self.model.phi_neg, feed_dict=_feed_dict)

    query_AP = []
    for q in range(n_queries):
      this_phi_pos = phi_pos[q][:num_pos[q]]
      this_phi_neg = phi_neg[q][:num_neg[q]]
      y_true = np.concatenate(
          (np.ones((this_phi_pos.shape[0])), np.zeros((this_phi_neg.shape[0]))),
          axis=0)
      y_scores = np.concatenate((this_phi_pos, this_phi_neg), axis=0)
      AP = self.apk(y_true, y_scores)
      query_AP.append(AP)
    mAP = sum(query_AP) / float(len(query_AP))
    return mAP
