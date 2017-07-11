from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import gzip
import numpy as np
import pickle as pkl

from scipy.misc import imread
from scipy.misc import imresize


class Dataset(object):

  def __init__(self, name, config, nway, split, batch_size, cache_path):
    self._name = name
    self._height = config.height
    self._width = config.width
    self._channels = config.channels
    self._config = config
    self._nway = nway
    self._split = split
    self._batch_size = batch_size
    self._cache_path = cache_path

    tr_dict, tr_label, val_dict, val_label, test_dict, test_label = self.read_dataset(
    )
    if self._split == "train":
      self._images_dict = tr_dict
      self._labels = tr_label
    elif self._split == "val":
      self._images_dict = val_dict
      self._labels = val_label
    elif self._split == "test":
      self._images_dict = test_dict
      self._labels = test_label

    self._num_examples = len(self._images_dict.keys())
    self._img_inds = np.arange(self._num_examples)
    self._classes = list(set(self._labels))
    self._num_classes = len(self._classes)

  def get_size(self):
    return self.num_examples

  def next_batch(self):
    if self.config.model_type == "siamese":
      batch_imgs, labels, inds_A_relative, inds_B_relative, pair_labels = self.get_batch_pairs(
      )
      batch = {
          "imgs": batch_imgs,
          "labels": labels,
          "inds_A": inds_A_relative,
          "inds_B": inds_B_relative,
          "pair_labels": pair_labels
      }
    elif self.config.model_type == "mAP":
      batch_imgs, selected_labels = self.get_batch_points()
      batch = {"imgs": batch_imgs, "labels": selected_labels}
    return batch

  def read_dataset(self):
    data = self.read_cache()
    if not data:
      train_dict, l_train, val_dict, l_val, test_dict, l_test = self.load_dataset(
      )
      self.save_cache(train_dict, l_train, val_dict, l_val, test_dict, l_test)
    else:
      train_dict = data["imgs_train_dict"]
      val_dict = data["imgs_val_dict"]
      test_dict = data["imgs_test_dict"]
      l_train = data["labels_train"]
      l_val = data["labels_val"]
      l_test = data["labels_test"]
    return train_dict, l_train, val_dict, l_val, test_dict, l_test

  def read_cache(self):
    """Reads dataset from cached pklz file."""
    print("Attempting to read from cache in {}...".format(self.cache_path))
    if os.path.exists(self.cache_path):
      with gzip.open(self.cache_path, "rb") as f:
        data = pkl.load(f)
      print("Successful.")
      return data
    else:
      print("Unsuccessful.")
      return False

  def save_cache(self, imgs_train_dict, labels_train, imgs_val_dict, labels_val,
                 images_test_dict, labels_test):
    """Saves pklz cache."""
    data = {
        "imgs_train_dict": imgs_train_dict,
        "labels_train": labels_train,
        "imgs_val_dict": imgs_val_dict,
        "labels_val": labels_val,
        "imgs_test_dict": images_test_dict,
        "labels_test": labels_test
    }
    with gzip.open(self.cache_path, "wb") as f:
      pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

  def load_img_as_array(self, img_loc):
    if self.channels == 3:
      _img_array = imread(img_loc, mode='RGB')
    else:
      _img_array = imread(img_loc, mode='L')
    img_array = imresize(
        _img_array, (self.height, self.width, self.channels), interp='bicubic')
    img_array = img_array.reshape((self.height, self.width, self.channels))
    return img_array

  def dense_to_one_hot(self, labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    flat_labels_one_hot = labels_one_hot.flatten()
    flat_dense = labels_dense.ravel()
    flat_labels_one_hot[index_offset + flat_dense.astype('int')] = 1
    return flat_labels_one_hot

  def load_batch_imgs(self, imgs_inds):
    raise NotImplemented

  def load_dataset(self):
    """
    Loads the dataset.
    :return:
    imgs_train_dict: A dictionary mapping the index of each training image to its location on disk
    labels_train: The label for each training image
    imgs_val_dict: A dictionary mapping the index of each validation image to its location on disk
    labels_val: The label for each validation image
    imgs_test_dict: A dictionary mapping the index of each test image to its location on disk
    labels_test: The label for each test image
    """
    raise NotImplemented

  def get_batch_points(self):
    """
    Construct a batch of self.batch_size many points
    belonging to self.nway different classes.
    """
    raise NotImplemented()

  def get_batch_pairs(self):
    """
    Create a batch for training the siamese network
    by selecting self.batch_size many points of self.nway classes
    and then forming all possible pairs from these points.
    """
    raise NotImplemented()

  def create_KshotNway_classification_episode(self, K, N, split):
    """
    Create an episode for K-shot N-way classification.

    :param K: Number of representatives of each class (to be used for classification of new points)
    :param N: Number of different classes
    :return:
      ref_paths: A list of N lists, each containing K paths
      query_paths: A list of N lists, each containing (20-K) paths
      labels: A list of the N class labels
    """
    raise NotImplemented

  def create_oneshotNway_retrieval_episode(self, N, n_per_class, split):
    """
    Sample a "pool" of images for 1-shot N-way retrieval.

    :param n_per_class: Number of sampled examples of each class in the "pool"
    :param N: Number of different classes
    :return:
      paths: The paths of each example in the pool
      labels: The corresponding class labels
    """
    raise NotImplemented

  @property
  def name(self):
    return self._name

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def img_inds(self):
    return self._img_inds

  @property
  def nway(self):
    return self._nway

  @property
  def config(self):
    return self._config

  @property
  def split(self):
    return self._split

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def height(self):
    return self._height

  @property
  def width(self):
    return self._width

  @property
  def channels(self):
    return self._channels

  @property
  def images_dict(self):
    return self._images_dict

  @property
  def labels(self):
    return self._labels

  @property
  def classes(self):
    return self._classes

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def cache_path(self):
    return self._cache_path
