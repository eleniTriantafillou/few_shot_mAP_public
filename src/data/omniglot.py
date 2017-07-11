from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
from itertools import combinations

from src.data.dataset import Dataset

OMNIGLOT_FOLDER = "data/omniglot/omniglot-master/python/"
OMNIGLOT_SPLITS_TRAINVAL = "data/dataset_splits/omniglot/trainval.txt"
OMNIGLOT_SPLITS_TRAIN = "data/dataset_splits/omniglot/train.txt"
OMNIGLOT_SPLITS_VAL = "data/dataset_splits/omniglot/val.txt"
OMNIGLOT_SPLITS_TEST = "data/dataset_splits/omniglot/test.txt"
OMNIGLOT_IMGS_BACKGROUND_ROT = OMNIGLOT_FOLDER + "images_background_resized_rot"
OMNIGLOT_IMGS_EVAL_ROT = OMNIGLOT_FOLDER + "images_evaluation_resized_rot"


class OmniglotDataset(Dataset):

  def __init__(self, name, config, nway, split, batch_size, cache_path):
    super(OmniglotDataset, self).__init__(name, config, nway, split, batch_size,
                                          cache_path)

  def get_batch_points_(self):
    char_classes = self.get_classes_list(self.split, augmented=False)
    n_classes_no_rot = len(char_classes)
    char_inds = np.random.choice(n_classes_no_rot, self.nway, replace=False)
    selected_chars = np.array(char_classes)[char_inds]
    rot_inds = np.random.choice(4, self.nway)
    angles = ['000', '090', '180', '270']
    selected_angles = [angles[i] for i in rot_inds]
    selected_classes = np.array([
        selected_chars[i] + '_rot_' + selected_angles[i]
        for i in range(self.nway)
    ])
    labels_no_rot = np.array([label[:-8] for label in list(self.labels)])
    rot_angles = np.array([label[-3:] for label in list(self.labels)])
    includable_point_inds = []
    for selected_char, selected_angle in zip(selected_chars, selected_angles):
      sat_char_inds = list(np.where(labels_no_rot == selected_char)[0])
      sat_inds = [
          ind for ind in sat_char_inds if rot_angles[ind] == selected_angle
      ]
      includable_point_inds += sat_inds

    num_includable_points = len(includable_point_inds)
    includable_point_inds_array = np.array(includable_point_inds)

    # Randomly select batch_size many of the 'allowed' examples
    selected_inds_inds = np.random.choice(
        num_includable_points, self.batch_size, replace=False)
    selected_inds = includable_point_inds_array[selected_inds_inds]

    selected_img_inds = self.img_inds[selected_inds]
    selected_labels = self.labels[selected_inds]
    batch_imgs = self.load_batch_imgs(selected_img_inds)
    return batch_imgs, selected_labels, selected_inds

  def get_batch_points(self):
    batch_imgs, selected_labels, selected_inds = self.get_batch_points_()
    return batch_imgs, selected_labels

  def get_batch_pairs(self):
    '''
    Forms all pairs from a given set of images
    :return: a batch for training the siamese network
    '''
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
    if self.config.loss_function == 'cross_entropy':
      pair_labels = self.dense_to_one_hot(pair_labels, 2)
      pair_labels = pair_labels.reshape(int(pair_labels.shape[0] / 2),
                                        2)  # for 2-way softmax
    return batch_imgs, labels, inds_A_relative, inds_B_relative, pair_labels

  def get_classes_list(self, split, augmented=True):

    def read_splits(fpath, augmented):
      classes = []
      with open(fpath, "r") as f:
        for line in f:
          if len(line) > 0:
            _class = line.strip()
            if augmented:
              for s in ['_rot_000', '_rot_090', '_rot_180', '_rot_270']:
                _class_rot = _class + s
                classes.append(_class_rot)
            else:
              classes.append(_class)
      return classes

    if split == "test":
      classes = read_splits(OMNIGLOT_SPLITS_TEST, augmented)
    elif split == "train" or split == "val":
      if split == "train":
        fpath = OMNIGLOT_SPLITS_TRAIN
      elif split == "val":
        fpath = OMNIGLOT_SPLITS_VAL

      # Check if trainval has already been split into train and val
      if os.path.exists(fpath):
        classes = read_splits(fpath, augmented)
      else:
        # Split trainval into train and val and write to disk
        trainval_classes_norot = read_splits(
            OMNIGLOT_SPLITS_TRAINVAL, augmented=False)
        num_trainval_classes = len(trainval_classes_norot)
        perm = np.arange(num_trainval_classes)
        np.random.shuffle(perm)
        num_training_classes = int(
            0.6 *
            num_trainval_classes)  # 60/40 split of trainval into train/val
        train_classes_inds = list(perm[:num_training_classes])
        val_classes_inds = list(perm[num_training_classes:])
        train_chars = np.array(trainval_classes_norot)[train_classes_inds]
        val_chars = np.array(trainval_classes_norot)[val_classes_inds]

        assert len(train_chars) + len(val_chars) == len(trainval_classes_norot), \
          "Num train chars {} + num val chars {} should be {}.".format(len(train_chars), len(val_chars),
                                                                       len(trainval_classes_norot))
        # Write to disk
        with open(OMNIGLOT_SPLITS_TRAIN, "a") as f:
          for c in train_chars:
            f.write("{}\n".format(c))
        with open(OMNIGLOT_SPLITS_VAL, "a") as f:
          for c in val_chars:
            f.write("{}\n".format(c))
        classes = read_splits(fpath, augmented)
    else:
      raise ValueError(
          "Unknown split. Please choose one of 'train', 'val' and 'test'.")
    return classes

  def load_batch_imgs(self, img_inds):
    batch_imgs = np.array([])
    for i in img_inds:
      img_path = self.images_dict[i]
      img_array = self.load_img_as_array(img_path)
      img_array = img_array.reshape((1, self._height, self._width,
                                     self._channels))
      if batch_imgs.shape[0] == 0:  # first image
        batch_imgs = img_array
      else:
        batch_imgs = np.concatenate((batch_imgs, img_array), axis=0)
    return batch_imgs

  def load_dataset(self):
    # Note: The trainval classes are split into training and validation classes
    # by randomly selecting 60 percent of the overall trainval *characters*
    # to be used for training (along with all 4 rotations of each)
    # and the remaining characters (with all their rotations) to be validation classes.
    root_folder_1 = OMNIGLOT_IMGS_BACKGROUND_ROT
    root_folder_2 = OMNIGLOT_IMGS_EVAL_ROT

    # dictionaries mapping index of image into the dataset
    # to the location of the image on disk
    imgs_train_dict = {}
    imgs_val_dict = {}
    imgs_test_dict = {}
    labels_train = []
    labels_val = []
    labels_test = []

    # For example, a class here is: Grantha/character08_rot180 if augmented is true
    train_classes = self.get_classes_list("train", augmented=True)
    val_classes = self.get_classes_list("val", augmented=True)
    test_classes = self.get_classes_list("test", augmented=True)

    example_ind_train = 0
    example_ind_val = 0
    example_ind_test = 0
    num_classes_loaded = -1
    for c in train_classes + val_classes + test_classes:
      num_classes_loaded += 1
      slash_ind = c.find('/')
      alphabet = c[:slash_ind]
      char = c[slash_ind + 1:]

      # Determine which folder this alphabet belongs to
      path1 = os.path.join(root_folder_1, alphabet)
      path2 = os.path.join(root_folder_2, alphabet)
      if os.path.isdir(path1):
        alphabet_folder = path1
      else:
        alphabet_folder = path2

      # The index of the example into the class (there are 20 of each class)
      class_image_num_train = 0
      class_image_num_val = 0
      class_image_num_test = 0

      # char is something like Grantha/character08_rot180
      underscore_ind = char.find('_')
      img_folder = os.path.join(alphabet_folder, char[:underscore_ind])
      rot_angle = char[-3:]  # one of '000', '090', '180', '270'
      img_files = [
          img_f for img_f in os.listdir(img_folder) if img_f[-7:-4] == rot_angle
      ]
      for img in img_files:
        img_loc = os.path.join(img_folder, img)
        if c in train_classes:
          imgs_train_dict[example_ind_train] = img_loc
          example_ind_train += 1
          label = c
          labels_train.append(label)
          class_image_num_train += 1
        elif c in val_classes:
          imgs_val_dict[example_ind_val] = img_loc
          example_ind_val += 1
          label = c
          labels_val.append(label)
          class_image_num_val += 1
        elif c in test_classes:
          imgs_test_dict[example_ind_test] = img_loc
          example_ind_test += 1
          label = c
          labels_test.append(label)
          class_image_num_test += 1
        else:
          raise ValueError("Found a class that does not belong to any split.")
    labels_train = np.array(labels_train)
    labels_val = np.array(labels_val)
    labels_test = np.array(labels_test)
    return imgs_train_dict, labels_train, imgs_val_dict, labels_val, imgs_test_dict, labels_test

  def create_KshotNway_classification_episode(self, K, N):
    all_classes = self.get_classes_list(self.split, augmented=True)
    num_classes = len(all_classes)
    perm = np.arange(num_classes)
    np.random.shuffle(perm)
    chosen_class_inds = list(perm[:N])

    ref_paths, query_paths, labels = [], [], []
    for n in range(N):
      c = all_classes[chosen_class_inds[n]]
      root_folder_1 = OMNIGLOT_IMGS_BACKGROUND_ROT
      root_folder_2 = OMNIGLOT_IMGS_EVAL_ROT
      slash_ind = c.find('/')
      alphabet = c[:slash_ind]
      char = c[slash_ind + 1:]

      # Determine which folder this alphabet belongs to
      # (since the new splits may have mixed which alphabets are background/evaluation)
      # with respect to these folders corresponding to the old splits.
      path1 = os.path.join(root_folder_1, alphabet)
      path2 = os.path.join(root_folder_2, alphabet)
      if os.path.isdir(path1):
        alphabet_folder = path1
      else:
        alphabet_folder = path2

      # char is something like Grantha/character08_rot180
      underscore_ind = char.find('_')
      img_folder = os.path.join(alphabet_folder, char[:underscore_ind])
      rot_angle = char[-3:]  # one of '000', '090', '180', '270'
      img_files = [
          img_f for img_f in os.listdir(img_folder) if img_f[-7:-4] == rot_angle
      ]

      # get an example of this character class
      img_example = img_files[0]  # for example 1040_06_rot_090.png
      char_baselabel = img_example[:img_example.find('_')]  # for example 1040

      # choose K images (drawers) to act as the representatives for this class
      drawer_inds = np.arange(20)
      np.random.shuffle(drawer_inds)
      ref_draw_inds = drawer_inds[:K]
      query_draw_inds = drawer_inds[K:]

      class_ref_paths = []
      class_query_paths = []
      for i in range(20):
        if len(str(i + 1)) < 2:
          str_ind = '0' + str(i + 1)
        else:
          str_ind = str(i + 1)
        img_name = char_baselabel + '_' + str_ind + '_rot_' + rot_angle + '.png'
        class_path = os.path.join(alphabet_folder, char[:underscore_ind])
        img_path = os.path.join(class_path, img_name)
        if i in ref_draw_inds:  # reference
          class_ref_paths.append(img_path)
        elif i in query_draw_inds:  # query
          class_query_paths.append(img_path)

      ref_paths.append(class_ref_paths)
      query_paths.append(class_query_paths)
      labels.append(n)
    return ref_paths, query_paths, labels

  def create_oneshotNway_retrieval_episode(self, N, n_per_class):
    all_classes = self.get_classes_list(self.split, augmented=True)
    num_classes = len(all_classes)
    perm = np.arange(num_classes)
    np.random.shuffle(perm)
    chosen_class_inds = list(perm[:N])

    paths, labels = [], []  # lists of length n_per_class * N
    for n in range(N):
      c = all_classes[chosen_class_inds[n]]
      root_folder_1 = OMNIGLOT_IMGS_BACKGROUND_ROT
      root_folder_2 = OMNIGLOT_IMGS_EVAL_ROT
      slash_ind = c.find('/')
      alphabet = c[:slash_ind]
      char = c[slash_ind + 1:]

      # Determine which folder this alphabet belongs to
      # (since the new splits may have mixed which alphabets are background/evaluation)
      # with respect to these folders corresponding to the old splits.
      path1 = os.path.join(root_folder_1, alphabet)
      path2 = os.path.join(root_folder_2, alphabet)
      if os.path.isdir(path1):
        alphabet_folder = path1
      else:
        alphabet_folder = path2

      # char is something like Grantha/character08_rot180
      underscore_ind = char.find('_')
      img_folder = os.path.join(alphabet_folder, char[:underscore_ind])
      rot_angle = char[-3:]  # one of '000', '090', '180', '270'
      img_files = [
          img_f for img_f in os.listdir(img_folder) if img_f[-7:-4] == rot_angle
      ]

      img_example = img_files[0]  # for example 1040_06_rot_090.png
      char_baselabel = img_example[:img_example.find('_')]  # for example 1040

      perm = np.arange(20)
      np.random.shuffle(perm)
      chosen_drawer_inds = list(perm[:n_per_class])
      for draw_ind, img in enumerate(img_files):
        if not draw_ind in chosen_drawer_inds:
          continue

        if len(str(draw_ind + 1)) < 2:
          str_ind = '0' + str(draw_ind + 1)
        else:
          str_ind = str(draw_ind + 1)

        img_name = char_baselabel + '_' + str_ind + '_rot_' + rot_angle + '.png'
        class_path = os.path.join(alphabet_folder, char[:underscore_ind])
        img_path = os.path.join(class_path, img_name)
        paths.append(img_path)
        labels.append(n)
    return paths, labels
