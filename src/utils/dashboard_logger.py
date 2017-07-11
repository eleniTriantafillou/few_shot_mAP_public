from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from time import strftime


class DashboardLogger(object):
  """
  Create the appropriate files to store
  results throughout training for use with
  deep dashboard.
  """

  def __init__(self, config):
    self._config = config
    self._name = config.name
    self._path = self.config.dashboard_path

    self._filenames_dict = {}
    self._filenames_dict["catalog"] = "catalog"
    self._filenames_dict["config"] = "config.txt"
    self._filenames_dict["notes"] = "notes.txt"
    self._filenames_dict["acc"] = ".csv"
    self._filenames_dict["F_score"] = ".csv"
    self._filenames_dict["1shot_5way_acc"] = "1shot_5way_acc.csv"
    self._filenames_dict["1shot_20way_acc"] = "1shot_20way_acc.csv"
    self._filenames_dict["5shot_5way_acc"] = "5shot_5way_acc.csv"
    self._filenames_dict["1shot_20way_mAP"] = "1shot_20way_mAP.csv"
    self._filenames_dict["1shot_5way_mAP"] = "1shot_5way_mAP.csv"
    self._filenames_dict["1shot_20way_mAP"] = "1shot_20way_mAP.csv"
    self._filenames_dict["mAP"] = "mAP.csv"

    self._paths_dict = self.create_paths()

    self.print_experiment_path()
    if not config.reload:
      self.setup()

  def print_experiment_path(self):
    print("Creating dashboard logger for experiment at location {}".format(
        self.path))

  def create_paths(self):
    paths_dict = {}
    for k, v in self.filenames_dict.iteritems():
      paths_dict[k] = os.path.join(self.path, v)
    return paths_dict

  def add_to_catalog(self, _type, fname, catalog_entry_name):
    print("Experiment path: {}".format(self.path))
    if not os.path.exists(self.paths_dict["catalog"]):
      with open(self.paths_dict["catalog"], "w+") as f:
        f.write("filename,type,name\n")
    with open(self.paths_dict["catalog"], "r") as f:
      lines = f.readlines()
    found = 0
    for line in lines:
      this_line = line.split(",")
      if this_line[0] == fname:
        found = 1
        break
    if not found:
      with open(self.paths_dict["catalog"], "a") as f:
        f.write('%s,%s,%s\n' % (fname, _type, catalog_entry_name))
    return

  def setup(self):
    if not os.path.isdir(self.path):
      os.makedirs(self.path)
    if self.config.model_type == "siamese":
      self.add_to_catalog("csv", self.filenames_dict["acc"],
                          'Accuracy During Training')
      with open(self.paths_dict["acc"], "w") as f:
        f.write("step,time,train acc,val acc\n")
    elif self.config.model_type == "mAP":
      self.add_to_catalog("csv", self.filenames_dict["F_score"],
                          "Scoring function")
      with open(self.paths_dict["F_score"], "w") as f:
        f.write("step,time,standard,augmented\n")
    if self.config.model_type == "mAP" or self.config.compute_mAP:
      self.add_to_catalog("csv", self.filenames_dict["mAP"],
                          "Mean Average Precision")
      with open(self.paths_dict["mAP"], "w") as f:
        f.write("step,time,train mAP,val mAP\n")
    if self.config.compute_fewshot:
      self.add_to_catalog("csv", self.filenames_dict["1shot_5way_acc"],
                          "1-shot 5-way Classification")
      with open(self.paths_dict["1shot_5way_acc"], "w") as f:
        f.write("step,time,acc\n")
      if self.config.dataset == "omniglot":
        self.add_to_catalog("csv", self.filenames_dict["1shot_20way_acc"],
                            "1-shot 20-way Classification")
        with open(self.paths_dict["1shot_20way_acc"], "w") as f:
          f.write("step,time,acc\n")
        self.add_to_catalog("csv", self.filenames_dict["1shot_20way_mAP"],
                            "1-shot 20-way Retrieval")
        with open(self.paths_dict["1shot_20way_mAP"], "w") as f:
          f.write("step,time,mAP\n")
      elif self.config.dataset == "mini_imagenet":
        if not self.config.model_type == "siamese":
          self.add_to_catalog("csv", self.filenames_dict["5shot_5way_acc"],
                              "5-shot 5-way Classification")
          with open(self.paths_dict["5shot_5way_acc"], "w") as f:
            f.write("step,time,acc\n")
        self.add_to_catalog("csv", self.filenames_dict["1shot_5way_mAP"],
                            "1-shot 5-way Retrieval")
        with open(self.paths_dict["1shot_5way_mAP"], "w") as f:
          f.write("step,time,mAP\n")
        self.add_to_catalog("csv", self.filenames_dict["1shot_20way_mAP"],
                            "1-shot 20-way Retrieval")
        with open(self.paths_dict["1shot_20way_mAP"], "w") as f:
          f.write("step,time,mAP\n")
    self.add_to_catalog("plain", self.filenames_dict["config"], "Config")
    self.write_config()
    self.add_to_catalog("plain", self.filenames_dict["notes"], "Notes")
    with open(self.paths_dict["notes"], "w") as f:
      f.write("Notes:\n")

  def write_acc(self, uidx, acc_train, acc_val):
    current_time = strftime("%Y-%m-%dT%H:%M:%S")
    with open(self.paths_dict["acc"], "a") as f:
      f.write("%d,%s,%f,%f\n" % (uidx, current_time, acc_train, acc_val))

  def write_F_score(self, uidx, standard, augmented):
    current_time = strftime("%Y-%m-%dT%H:%M:%S")
    with open(self.paths_dict["acc"], "a") as f:
      f.write("%d,%s,%f,%f\n" % (uidx, current_time, standard, augmented))

  def write_Kshot_Nway_classif(self, uidx, K, N, acc):
    current_time = strftime("%Y-%m-%dT%H:%M:%S")
    key = "{}shot_{}way_acc".format(K, N)
    with open(self.paths_dict[key], "a") as f:
      f.write("%d,%s,%f\n" % (uidx, current_time, acc))

  def write_oneshot_Nway_retrieval(self, uidx, N, mAP):
    key = "1shot_{}way_mAP".format(N)
    current_time = strftime("%Y-%m-%dT%H:%M:%S")
    with open(self.paths_dict[key], "a") as f:
      f.write("%d,%s,%f\n" % (uidx, current_time, mAP))

  def write_mAP(self, uidx, train_mAP, val_mAP):
    current_time = strftime("%Y-%m-%dT%H:%M:%S")
    with open(self.paths_dict["mAP"], "a") as f:
      f.write("%d,%s,%f,%f\n" % (uidx, current_time, train_mAP, val_mAP))

  def write_config(self):
    with open(self.paths_dict["config"], "w") as f:
      attr_dict = vars(self.config)
      for key, val in attr_dict.iteritems():
        f.write("{}: {}\n".format(key, val))

  def take_note(self, note):
    with open(self.paths_dict["notes"], "a") as f:
      f.write("{}".format(note))

  @property
  def name(self):
    return self._name

  @property
  def config(self):
    return self._config

  @property
  def path(self):
    return self._path

  @property
  def path_notes(self):
    return self._path_notes

  @property
  def path_acc(self):
    return self._path_acc

  @property
  def path_catalog(self):
    return self._path_catalog

  @property
  def paths_dict(self):
    return self._paths_dict

  @property
  def filenames_dict(self):
    return self._filenames_dict
