from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from src.data.omniglot import OmniglotDataset
from src.data.mini_imagenet import MiniImageNetDataset


def get_dataset(dataset_name, config, split):
  if dataset_name == "omniglot":
    return OmniglotDataset(dataset_name, config, config.nway, split,
                           config.batch_size, "data/omniglot_cache.pklz")
  elif dataset_name == "mini_imagenet":
    return MiniImageNetDataset(dataset_name, config, config.nway, split,
                               config.batch_size,
                               "data/mini_imagenet_cache.pklz")
  else:
    raise ValueError("Unknown dataset \"{}\"".format(dataset_name))
