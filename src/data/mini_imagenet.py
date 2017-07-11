from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from tqdm import tqdm
import csv
import numpy as np
from itertools import combinations

from src.data.dataset import Dataset

MINI_IMAGENET_FOLDER = "data/mini_imagenet/"
MINI_IMAGENET_SPLITS_FOLDER = "data/dataset_splits/mini_imagenet/Ravi"


class MiniImageNetDataset(Dataset):

  def __init__(self, name, config, nway, split, batch_size, cache_path):
    self._cache_path = cache_path
    super(MiniImageNetDataset, self).__init__(name, config, nway, split,
                                              batch_size, cache_path)

  def get_batch_points(self):
    batch_imgs, selected_labels, selected_inds = self.get_batch_points_()
    return batch_imgs, selected_labels

  def get_batch_points_(self):
    class_inds = np.random.choice(self.num_classes, self.nway, replace=False)
    selected_classes = np.array(self.classes)[class_inds]

    includable_point_inds = []
    for selected_class in selected_classes:
      sat_inds = list(np.where(self.labels == selected_class)[0])
      includable_point_inds += sat_inds
    num_includable_points = len(includable_point_inds)
    includable_point_inds_array = np.array(includable_point_inds)

    # Now randomly select batch_size many of the "allowed" examples
    selected_inds_ = np.random.choice(
        num_includable_points, self.batch_size, replace=False)
    selected_inds = includable_point_inds_array[selected_inds_]

    selected_labels = self.labels[selected_inds]
    batch_imgs = self.load_batch_imgs(selected_inds)
    return batch_imgs, selected_labels, selected_inds

  def get_batch_pairs(self):
    batch_imgs, labels, selected_inds = self.get_batch_points_()
    relative_inds = np.arange(self.batch_size)
    pair_inds = [list(x) for x in combinations(relative_inds, 2)]
    label1s = [labels[pair[0]] for pair in pair_inds]
    label2s = [labels[pair[1]] for pair in pair_inds]
    _pair_labels = [
        int(label1s[i] == label2s[i]) for i in range(len(pair_inds))
    ]
    inds_A_relative = [pair[0] for pair in pair_inds]
    inds_B_relative = [pair[1] for pair in pair_inds]
    pair_labels = np.array(_pair_labels)
    pair_labels = pair_labels.reshape(pair_labels.shape[0], 1)

    if self.config.loss_function == "cross_entropy":
      pair_labels = self.dense_to_one_hot(pair_labels, 2)
      pair_labels = pair_labels.reshape(int(pair_labels.shape[0] / 2), 2)
    return batch_imgs, labels, inds_A_relative, inds_B_relative, pair_labels

  def read_splits(self, split):
    classes, files = [], []
    csv_path = os.path.join(MINI_IMAGENET_SPLITS_FOLDER, split + ".csv")
    with open(csv_path, "r") as csvfile:
      csvreader = csv.reader(csvfile)
      for i, row in enumerate(csvreader):
        if i == 0:
          continue
        if len(row[0]) == 0:
          break
        files.append(row[0])
        classes.append(row[1])
    unique_classes = list(set(classes))
    return files, unique_classes

  def load_batch_imgs(self, img_inds):
    batch_imgs = np.array([])
    for i in img_inds:
      img_path = self.images_dict[i]
      img_array = self.load_img_as_array(img_path)
      img_array = img_array.reshape((1, self._height, self._width,
                                     self._channels))
      if img_array.shape[2] == 1:  # black-and-white images
        img_array = np.zeros((self._height, self._width, self._channels))
        img_array[:, :, 0] = img_array.reshape((self._height, self._width))
        img_array[:, :, 1] = img_array.reshape((self._height, self._width))
        img_array[:, :, 2] = img_array.reshape((self._height, self._width))
      else:
        img_array = img_array

      img_array = img_array.reshape((1, self._height, self._width,
                                     self._channels))
      if batch_imgs.shape[0] == 0:
        batch_imgs = img_array
      else:
        batch_imgs = np.concatenate((batch_imgs, img_array), axis=0)
    return batch_imgs

  def load_dataset(self):
    imgs_dict_train, imgs_dict_val, imgs_dict_test = {}, {}, {}
    labels_train, labels_val, labels_test = [], [], []
    train_files, train_classes = self.read_splits("train")
    val_files, val_classes = self.read_splits("val")
    test_files, test_classes = self.read_splits("test")
    classes = train_classes + val_classes + test_classes
    all_imgs = os.listdir(MINI_IMAGENET_FOLDER)

    example_ind_train, example_ind_val, example_ind_test = 0, 0, 0
    for idx in tqdm(range(len(classes)), desc='Loading mini-ImageNet'):
      c = classes[idx]
      class_imgs = [f for f in all_imgs if c in f]
      assert len(class_imgs) == 600, "Expected 600 images but found {}".format(
          len(class_imgs))

      for i in range(len(class_imgs)):
        img = class_imgs[i]
        img_path = os.path.join(MINI_IMAGENET_FOLDER, img)
        if img in train_files and c in train_classes:
          imgs_dict_train[example_ind_train] = img_path
          labels_train.append(c)
          example_ind_train += 1
        elif img in val_files and c in val_classes:
          imgs_dict_val[example_ind_val] = img_path
          labels_val.append(c)
          example_ind_val += 1
        elif img in test_files and c in test_classes:
          imgs_dict_test[example_ind_test] = img_path
          labels_test.append(c)
          example_ind_test += 1
        else:
          raise ValueError("Found an image that does not belong to any split.")

    labels_train = np.array(labels_train)
    labels_val = np.array(labels_val)
    labels_test = np.array(labels_test)

    assert len(labels_train
              ) == 600 * 64, "Expected {} but found {} train examples.".format(
                  600 * 64, len(labels_train))
    assert len(
        labels_val
    ) == 600 * 16, "Expected {} but found {} validation examples.".format(
        600 * 16, len(labels_val))
    assert len(labels_test
              ) == 600 * 20, "Expected {} but found {} test examples.".format(
                  600 * 20, len(labels_test))
    return imgs_dict_train, labels_train, imgs_dict_val, labels_val, imgs_dict_test, labels_test

  def create_KshotNway_classification_episode(self, K, N):
    files, all_classes = self.read_splits(self.split)
    num_classes = len(all_classes)
    perm = np.arange(num_classes)
    np.random.shuffle(perm)
    chosen_class_inds = list(perm[:N])
    chosen_classes = np.array(all_classes)[chosen_class_inds]

    ref_paths, query_paths, labels = [], [], []  # Lists of lists
    all_imgs = os.listdir(MINI_IMAGENET_FOLDER)
    for class_ind, c in enumerate(chosen_classes):
      class_imgs = [
          os.path.join(MINI_IMAGENET_FOLDER, f) for f in all_imgs if c in f
      ]
      num_examples = len(class_imgs)
      perm = np.arange(num_examples)
      np.random.shuffle(perm)
      chosen_examples_inds = list(perm[:20])
      chosen_examples = np.array(class_imgs)[chosen_examples_inds]

      class_ref_paths = []
      class_query_paths = []
      for i, img in enumerate(chosen_examples):
        if i < K:  # Let the first K be representatives
          class_ref_paths.append(img)
        else:
          class_query_paths.append(img)
      ref_paths.append(class_ref_paths)
      query_paths.append(class_query_paths)
      labels.append(class_ind)
    return ref_paths, query_paths, labels

  def create_oneshotNway_retrieval_episode(self, N, n_per_class):
    files, all_classes = self.read_splits(self.split)
    num_classes = len(all_classes)
    perm = np.arange(num_classes)
    np.random.shuffle(perm)
    chosen_class_inds = list(perm[:N])
    chosen_classes = np.array(all_classes)[chosen_class_inds]

    paths, labels = [], []  # lists of length n_per_class * N
    all_imgs = os.listdir(MINI_IMAGENET_FOLDER)
    for class_ind, c in enumerate(chosen_classes):
      class_imgs = [
          os.path.join(MINI_IMAGENET_FOLDER, f) for f in all_imgs if c in f
      ]
      num_examples = len(class_imgs)
      perm = np.arange(num_examples)
      np.random.shuffle(perm)
      chosen_examples_inds = list(perm[:n_per_class])
      chosen_examples = np.array(class_imgs)[chosen_examples_inds]
      for img in chosen_examples:
        paths.append(img)
        labels.append(class_ind)
    return paths, labels
