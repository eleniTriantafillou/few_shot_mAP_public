from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Model(object):
  """
  Model class.
  """

  def __init__(self, config, reuse=False):
    self._config = config
    self._x = tf.placeholder(
        tf.float32, [None, config.height, config.width, config.channels],
        name="x")

    embedding = self.forward_pass(reuse)
    self._feats = tf.truediv(
        embedding,
        tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True)))

    # Number of relevant points for each query
    self._num_pos = tf.placeholder(tf.int32, [None], name="num_pos")
    self._num_neg = tf.placeholder(tf.int32, [None], name="num_neg")
    self._batch_size = tf.shape(self._x)[0]

    # The inds belonging to the positive and negative sets for each query
    self._pos_inds = tf.placeholder(tf.int32, [None, None], name="pos_inds")
    self._neg_inds = tf.placeholder(tf.int32, [None, None], name="neg_inds")

    self._n_queries_to_parse = tf.placeholder(
        tf.int32, [], name="n_queries_to_parse")

    # The solution of loss-augmented inference for each query
    self._Y_aug = tf.placeholder(
        tf.float32, [None, None, None],
        name="Y_aug")  # (num queries, num_pos, num_neg)

    self._phi_pos, self._phi_neg, self._mAP_score_std, \
    self._mAP_score_aug, self._mAP_score_GT, self._skipped_queries = self.perform_inference_mAP()
    self._loss = self.compute_loss()
    self._train_step = self.get_train_step()

  def forward_pass(self, reuse, _print=True):
    """
    Perform a forward pass through the network

    :param reuse: Whether to re-use the network's parameters.
    :param _print: Whether to print the shapes of activations.
    :return:
    """

    def print_activations(t, _print=True):
      if _print:
        print(t.op.name, ' ', t.get_shape().as_list())

    conv1 = slim.convolution2d(
        self.x,
        64,
        kernel_size=[3, 3],
        activation_fn=slim.nn.relu,
        normalizer_fn=slim.batch_norm,
        scope='conv1',
        reuse=reuse)
    print_activations(conv1, _print=_print)
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='VALID', scope='pool1')
    print_activations(pool1, _print=_print)

    conv2 = slim.convolution2d(
        pool1,
        64,
        kernel_size=[3, 3],
        activation_fn=slim.nn.relu,
        normalizer_fn=slim.batch_norm,
        scope='conv2',
        reuse=reuse)
    print_activations(conv2, _print=_print)
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='VALID', scope='pool2')
    print_activations(pool2, _print=_print)

    conv3 = slim.convolution2d(
        pool2,
        64,
        kernel_size=[3, 3],
        activation_fn=slim.nn.relu,
        normalizer_fn=slim.batch_norm,
        scope='conv3',
        reuse=reuse)
    print_activations(conv3, _print=_print)
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='VALID', scope='pool3')
    print_activations(pool3, _print=_print)

    conv4 = slim.convolution2d(
        pool3,
        64,
        kernel_size=[3, 3],
        activation_fn=slim.nn.relu,
        normalizer_fn=slim.batch_norm,
        scope='conv4',
        reuse=reuse)
    print_activations(conv4, _print=_print)
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='VALID', scope='pool4')
    print_activations(pool4, _print=_print)

    all_but_first_dims = pool4.get_shape().as_list()[1:]
    mult_dims = 1
    for dim in all_but_first_dims:
      mult_dims = mult_dims * dim
    embedding = tf.reshape(pool4, [-1, mult_dims])
    print_activations(embedding, _print=_print)

    return embedding

  def get_train_step(self):
    lr = tf.get_variable(
        "learning_rate",
        shape=[],
        initializer=tf.constant_initializer(self.config.lr),
        dtype=tf.float32,
        trainable=False)
    if self.config.optimizer == 'ADAM':
      optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    elif self.config.optimizer == 'SGD':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif self.config.optimizer == 'SGD_momentum':
      optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    train_step = optimizer.minimize(self.loss)
    self._lr = lr
    self._new_lr = tf.placeholder(tf.float32, [], name="new_lr")
    self._assign_lr = tf.assign(self._lr, self._new_lr)
    return train_step

  def compute_siamese_accuracy(self):
    raise NotImplemented

  def get_positive_negative_splits(self, batch):
    """
    Compute the positive/negative sets for each query in the batch.

    :param batch: A batch of points
    :return: num_pos: An array of the number of positive points of each query
    :return: num_neg: An array of the number of negative points of each query
    :return: pos_inds: An array of the inds of positive points of each query
    :return: neg_inds: An array of the inds of negative points of each query
    """

    pos_inds, neg_inds = [], []  # Lists of lists
    num_pos, num_neg = [], []
    batch_labels = batch["labels"]
    batch_size = len(batch_labels)

    def get_query_splits(query_ind):
      """
      Create the splits for a single query.

      :param query_ind: Index of the query in the batch
      :return: q_pos_inds: The inds of the query's positive points
      :return: q_neg_inds: The inds of the query's negative points
      """

      query_label = batch_labels[query_ind]
      q_pos_inds, q_neg_inds = [], []
      for i in range(batch_size):
        if i == query_ind:
          continue
        this_label = batch_labels[i]
        if this_label == query_label:
          q_pos_inds.append(i)
        else:  # negative point
          q_neg_inds.append(i)
      return q_pos_inds, q_neg_inds

    for i in range(batch_size):
      q_pos_inds, q_neg_inds = get_query_splits(i)
      pos_inds.append(q_pos_inds)
      neg_inds.append(q_neg_inds)
      num_pos.append(len(q_pos_inds))
      num_neg.append(len(q_neg_inds))

    # Pad all lists to be max length and make np arrays
    max_pos, max_neg = max(num_pos), max(num_neg)
    pos_inds_np = np.zeros((batch_size, max_pos))
    neg_inds_np = np.zeros((batch_size, max_neg))
    for i in range(batch_size):
      this_len_pos = len(pos_inds[i])
      pos_inds_np[i, :this_len_pos] = pos_inds[i]
      this_len_neg = len(neg_inds[i])
      neg_inds_np[i, :this_len_neg] = neg_inds[i]
    num_pos, num_neg = np.array(num_pos), np.array(num_neg)
    pos_inds, neg_inds = pos_inds_np, neg_inds_np
    return [num_pos, num_neg, pos_inds, neg_inds]

  def perform_query_inference(self, q_feats, q_pos_feats, q_neg_feats,
                              q_num_pos, q_num_neg, q_Y_aug):
    """
    Inference for a specific query.

    :param q_feats: the features for the query
    :param q_pos_feats: the features of the query's positive points
    :param q_neg_feats: the features of the query's negative points
    :param q_num_pos: the number of positive points for the query
    :param q_num_neg: the number of negative points for the query
    :param q_Y_aug: the solution of loss-augmented inference for this query

    :return: phi_pos: the similarity between the query and each positive point
    :return: phi_neg: the similarity between the query and each negative point
    :return: AP_score_std: the score of the standard inference solution for AP of this query
    :return: AP_score_aug: the score of the loss-augmented inference solution for AP of this query
    :return: AP_score_GT: the score of the ground truth solution for AP of this query
    """

    S_pos = tf.matmul(q_feats, q_pos_feats, transpose_b=True)  # (1, num_pos)
    S_neg = tf.matmul(q_feats, q_neg_feats, transpose_b=True)  # (1, num_neg)
    phi_pos, sorted_inds_pos = tf.nn.top_k(S_pos, k=q_num_pos)
    phi_neg, sorted_inds_neg = tf.nn.top_k(S_neg, k=q_num_neg)

    phi_pos = tf.transpose(phi_pos)
    phi_neg = tf.transpose(phi_neg)

    # Score of standard inference
    phi_pos_expanded = tf.tile(phi_pos, [1, q_num_neg])  # (num_pos, num_neg)
    phi_neg_expanded = tf.tile(tf.transpose(phi_neg), [q_num_pos,
                                                       1])  # (num_pos, num_neg)
    temp1_Y = tf.greater(phi_pos_expanded,
                         phi_neg_expanded)  # (num_pos, num_neg) of True/False's
    temp2_Y = 2. * tf.to_float(temp1_Y)  # (num_pos, num_neg) of 2/0's
    Y_std = temp2_Y - tf.ones_like(temp2_Y)  # (num_pos, num_neg) of 1/-1's
    F_std = Y_std * (phi_pos_expanded - phi_neg_expanded)  # (num_pos, num_neg)
    AP_score_std = tf.truediv(
        tf.reduce_sum(F_std), tf.to_float(q_num_pos * q_num_neg))

    # Score of loss-augmented inferred ranking
    F_aug = q_Y_aug * (phi_pos_expanded - phi_neg_expanded)
    AP_score_aug = tf.truediv(
        tf.reduce_sum(F_aug), tf.to_float(q_num_pos * q_num_neg))

    # Score of the groundtruth
    q_Y_GT = tf.ones_like(Y_std)
    F_GT = q_Y_GT * (phi_pos_expanded - phi_neg_expanded)
    AP_score_GT = tf.truediv(
        tf.reduce_sum(F_GT), tf.to_float(q_num_pos * q_num_neg))

    AP_score_std = tf.reshape(AP_score_std, [1, 1])
    AP_score_aug = tf.reshape(AP_score_aug, [1, 1])
    AP_score_GT = tf.reshape(AP_score_GT, [1, 1])
    return phi_pos, phi_neg, AP_score_std, AP_score_aug, AP_score_GT

  def perform_inference_mAP(self):
    """
    Perform inference for the task loss of mAP.
    This involves looping over the different queries
    in the batch to compute the AP of each and then
    composing these scores.

    """

    def body(i, skipped_q_prev, score_std_prev, score_aug_prev, score_GT_prev,
             phi_pos_prev, phi_neg_prev):
      """

      :param i: The index of the query currently being considered
      :param skipped_q_prev: Binary array indicating whether queries were skipped, up till the ith
      :param score_std_prev: Score of standard inference solution for queries up till the ith
      :param score_aug_prev: Score of loss-augmented inference solution for queries up till the ith
      :param score_GT_prev: Score of ground truth solution for queries up till the ith
      :param phi_pos_prev: Cosine similarities between each of the first (i-1) queries and their positives
      :param phi_neg_prev: Cosine similarities between each of the first (i-1) queries and their negatives

      :return: The same quantities but after having incorporated the ith query as well.

      """

      query_feats = tf.reshape(tf.gather(self.feats, i), [1, -1])
      pos_feats_inds = tf.gather(self.pos_inds, i)
      pos_feats = tf.gather(self.feats, pos_feats_inds)
      neg_feats_inds = tf.gather(self.neg_inds, i)
      neg_feats = tf.gather(self.feats, neg_feats_inds)
      q_num_pos = tf.gather(self.num_pos, i)
      q_num_neg = tf.gather(self.num_neg, i)
      _q_Y_aug = tf.gather(self.Y_aug, i)

      max_pos = tf.reduce_max(self.num_pos)
      max_neg = tf.reduce_max(self.num_neg)

      q_Y_aug = tf.slice(_q_Y_aug, [0, 0], [q_num_pos, q_num_neg])

      # Case where the ith point in the batch forms a non-empty positive set
      def _use_query():
        q_phi_pos, q_phi_neg, q_score_std, q_score_aug, q_score_GT = self.perform_query_inference(
            query_feats, pos_feats, neg_feats, q_num_pos, q_num_neg, q_Y_aug)

        # In what follows, we update score_std_next, score_aug_next
        # Note that due to the requirement of quantities score_std_next, score_aug_next, etc to have
        # the same shape in each iteration, we zero-pad them to be batch_size-long.
        def _first_score():
          this_score_std_padded = tf.concat(
              [q_score_std, tf.zeros([self.batch_size - i - 1, 1])], 0)
          this_score_aug_padded = tf.concat(
              [q_score_aug, tf.zeros([self.batch_size - i - 1, 1])], 0)
          this_score_GT_padded = tf.concat(
              [q_score_GT, tf.zeros([self.batch_size - i - 1, 1])], 0)
          score_std_next = tf.add(score_std_prev, this_score_std_padded)
          score_aug_next = tf.add(score_aug_prev, this_score_aug_padded)
          score_GT_next = tf.add(score_GT_prev, this_score_GT_padded)
          return score_std_next, score_aug_next, score_GT_next

        def _else_score():
          temp = tf.concat([tf.zeros([i, 1]), q_score_std], 0)
          this_score_std_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, 1])], 0)
          temp = tf.concat([tf.zeros([i, 1]), q_score_aug], 0)
          this_score_aug_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, 1])], 0)
          temp = tf.concat([tf.zeros([i, 1]), q_score_GT], 0)
          this_score_GT_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, 1])], 0)
          score_std_next = tf.add(score_std_prev, this_score_std_padded)
          score_aug_next = tf.add(score_aug_prev, this_score_aug_padded)
          score_GT_next = tf.add(score_GT_prev, this_score_GT_padded)
          return score_std_next, score_aug_next, score_GT_next

        score_std_next, score_aug_next, score_GT_next = tf.cond(
            tf.equal(i, tf.constant(0)), _first_score, _else_score)

        q_phi_pos = tf.transpose(q_phi_pos)
        q_phi_neg = tf.transpose(q_phi_neg)

        # Concatenate with appropriate amount of zeros, to pad along each row
        # This padding is different from above: we're not padding along the dimension
        # of queries in order to get to batch_size, but we're padding the phi_pos of
        # just the single query to get to max_pos
        # (since not all queries will have the same number of positive points)
        padding_phi_pos = tf.zeros([1, max_pos - q_num_pos])
        padding_phi_neg = tf.zeros([1, max_neg - q_num_neg])
        this_phi_pos = tf.concat([q_phi_pos, padding_phi_pos], 1)
        this_phi_neg = tf.concat([q_phi_neg, padding_phi_neg], 1)

        # Update phi_pos_next, phi_neg_next
        def _first_phi():
          this_phi_pos_padded = tf.concat(
              [this_phi_pos, tf.zeros([self.batch_size - i - 1, max_pos])], 0)
          this_phi_neg_padded = tf.concat(
              [this_phi_neg, tf.zeros([self.batch_size - i - 1, max_neg])], 0)
          phi_pos_next = tf.add(phi_pos_prev, this_phi_pos_padded)
          phi_neg_next = tf.add(phi_neg_prev, this_phi_neg_padded)
          return phi_pos_next, phi_neg_next

        def _else_phi():
          temp = tf.concat([tf.zeros([i, max_pos]), this_phi_pos], 0)
          this_phi_pos_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, max_pos])], 0)
          temp = tf.concat([tf.zeros([i, max_neg]), this_phi_neg], 0)
          this_phi_neg_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, max_neg])], 0)
          phi_pos_next = tf.add(phi_pos_prev, this_phi_pos_padded)
          phi_neg_next = tf.add(phi_neg_prev, this_phi_neg_padded)
          return phi_pos_next, phi_neg_next

        phi_pos_next, phi_neg_next = tf.cond(
            tf.equal(i, tf.constant(0)), _first_phi, _else_phi)

        # The appropriate entry is already 0 which we keep since we didn't skip this query
        skipped_q_next = skipped_q_prev
        return score_std_next, score_aug_next, score_GT_next, phi_pos_next, phi_neg_next, skipped_q_next

      def _dont_use_query():

        q_score_std = tf.zeros([1, 1])
        q_score_aug = tf.zeros([1, 1])
        q_score_GT = tf.zeros([1, 1])

        # Update score_std_next, score_aug_next
        def _first_score():
          this_score_std_padded = tf.concat(
              [q_score_std, tf.zeros([self.batch_size - i - 1, 1])], 0)
          this_score_aug_padded = tf.concat(
              [q_score_aug, tf.zeros([self.batch_size - i - 1, 1])], 0)
          this_score_GT_padded = tf.concat(
              [q_score_GT, tf.zeros([self.batch_size - i - 1, 1])], 0)
          score_std_next = tf.add(score_std_prev, this_score_std_padded)
          score_aug_next = tf.add(score_aug_prev, this_score_aug_padded)
          score_GT_next = tf.add(score_GT_prev, this_score_GT_padded)
          return score_std_next, score_aug_next, score_GT_next

        def _else_score():
          temp = tf.concat([tf.zeros([i, 1]), q_score_std], 0)
          this_score_std_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, 1])], 0)
          temp = tf.concat([tf.zeros([i, 1]), q_score_aug], 0)
          this_score_aug_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, 1])], 0)
          temp = tf.concat([tf.zeros([i, 1]), q_score_GT], 0)
          this_score_GT_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, 1])], 0)
          score_std_next = tf.add(score_std_prev, this_score_std_padded)
          score_aug_next = tf.add(score_aug_prev, this_score_aug_padded)
          score_GT_next = tf.add(score_GT_prev, this_score_GT_padded)
          return score_std_next, score_aug_next, score_GT_next

        score_std_next, score_aug_next, score_GT_next = tf.cond(
            tf.equal(i, tf.constant(0)), _first_score, _else_score)

        q_phi_pos = tf.zeros([1, max_pos])
        q_phi_neg = tf.zeros([1, max_neg])

        # Update phi_pos_next, phi_neg_next
        def _first_phi():
          this_phi_pos_padded = tf.concat(
              [q_phi_pos, tf.zeros([self.batch_size - i - 1, max_pos])], 0)
          this_phi_neg_padded = tf.concat(
              [q_phi_neg, tf.zeros([self.batch_size - i - 1, max_neg])], 0)
          phi_pos_next = tf.add(phi_pos_prev, this_phi_pos_padded)
          phi_neg_next = tf.add(phi_neg_prev, this_phi_neg_padded)
          return phi_pos_next, phi_neg_next

        def _else_phi():
          temp = tf.concat([tf.zeros([i, max_pos]), q_phi_pos], 0)
          this_phi_pos_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, max_pos])], 0)
          temp = tf.concat([tf.zeros([i, max_neg]), q_phi_neg], 0)
          this_phi_neg_padded = tf.concat(
              [temp, tf.zeros([self.batch_size - i - 1, max_neg])], 0)
          phi_pos_next = tf.add(phi_pos_prev, this_phi_pos_padded)
          phi_neg_next = tf.add(phi_neg_prev, this_phi_neg_padded)
          return phi_pos_next, phi_neg_next

        phi_pos_next, phi_neg_next = tf.cond(
            tf.equal(i, tf.constant(0)), _first_phi, _else_phi)

        # Update skipped_q_next
        # make ith position a 1 in the one-hot-encoded vector
        def _first_skip():
          skipped_this = tf.concat(
              [tf.ones([1]), tf.zeros([self.batch_size - i - 1])], 0)
          skipped_q_next = tf.add(skipped_q_prev, skipped_this)
          return skipped_q_next

        def _else_skip():
          temp = tf.concat([tf.zeros([i]), tf.ones([1])], 0)
          skipped_this = tf.concat([temp, tf.zeros([self.batch_size - i - 1])], 0)
          skipped_q_next = tf.add(skipped_q_prev, skipped_this)
          return skipped_q_next

        skipped_q_next = tf.cond(
            tf.equal(i, tf.constant(0)), _first_skip, _else_skip)
        return score_std_next, score_aug_next, score_GT_next, phi_pos_next, phi_neg_next, skipped_q_next

      _use_query_cond = tf.greater(q_num_pos, 0)
      score_std_next, score_aug_next, score_GT_next, phi_pos_next, phi_neg_next, skipped_q_next = tf.cond(
          _use_query_cond, _use_query, _dont_use_query)
      i = tf.add(i, 1)
      return i, skipped_q_next, score_std_next, score_aug_next, score_GT_next, phi_pos_next, phi_neg_next

    i = tf.constant(0)

    def condition(i, _1, _2, _3, _4, _5, _6):
      return tf.less(i, self.n_queries_to_parse)

    # Initialize the loop variables - their size will remain unchanged throughout the loop.
    phi_pos = tf.zeros([self.batch_size, tf.reduce_max(self.num_pos)])
    phi_neg = tf.zeros([self.batch_size, tf.reduce_max(self.num_neg)])
    score_std = tf.zeros([self.batch_size, 1])
    score_aug = tf.zeros([self.batch_size, 1])
    score_GT = tf.zeros([self.batch_size, 1])
    # One-hot-encoded vector indicating which queries were skipped
    # (a query is skipped if it can't create a positive set)
    skipped_queries = tf.zeros([self.batch_size])

    _i, skipped_queries, score_std, score_aug, score_GT, phi_pos, phi_neg = tf.while_loop(
        condition,
        body,
        loop_vars=[
            i, skipped_queries, score_std, score_aug, score_GT, phi_pos, phi_neg
        ])
    mAP_score_std = tf.reduce_mean(score_std)
    mAP_score_aug = tf.reduce_mean(score_aug)
    mAP_score_GT = tf.reduce_mean(score_GT)
    return phi_pos, phi_neg, mAP_score_std, mAP_score_aug, mAP_score_GT, skipped_queries

  @property
  def train_step(self):
    return self._train_step

  @property
  def x(self):
    return self._x

  @property
  def config(self):
    return self._config

  @property
  def feats(self):
    return self._feats

  @property
  def lr(self):
    return self._lr

  @property
  def new_lr(self):
    return self._new_lr

  @property
  def assign_lr(self):
    return self._assign_lr

  @property
  def loss(self):
    return self._loss

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_pos(self):
    return self._num_pos

  @property
  def num_neg(self):
    return self._num_neg

  @property
  def pos_inds(self):
    return self._pos_inds

  @property
  def neg_inds(self):
    return self._neg_inds

  @property
  def Y_aug(self):
    return self._Y_aug

  @property
  def phi_pos(self):
    return self._phi_pos

  @property
  def phi_neg(self):
    return self._phi_neg

  @property
  def mAP_score_std(self):
    return self._mAP_score_std

  @property
  def mAP_score_aug(self):
    return self._mAP_score_aug

  @property
  def mAP_score_GT(self):
    return self._mAP_score_GT

  @property
  def skipped_queries(self):
    return self._skipped_queries

  @property
  def n_queries_to_parse(self):
    return self._n_queries_to_parse
