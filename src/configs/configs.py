from src.configs.config_siamese_omniglot import SiameseConfigOmniglot
from src.configs.config_mAP_omniglot import MeanAveragePrecisionConfigOmniglot
from src.configs.config_siamese_mini_imagenet import SiameseConfigMiniImageNet
from src.configs.config_mAP_mini_imagenet import MeanAveragePrecisionConfigMiniImageNet


class Configs(object):

  def __init__(self):
    self._CONFIGS = {}
    self._CONFIGS["omniglot_siamese"] = SiameseConfigOmniglot()
    self._CONFIGS["omniglot_mAP"] = MeanAveragePrecisionConfigOmniglot()
    self._CONFIGS["mini_imagenet_siamese"] = SiameseConfigMiniImageNet()
    self._CONFIGS[
        "mini_imagenet_mAP"] = MeanAveragePrecisionConfigMiniImageNet()

  def get_config(self, dataset_name, model_name):
    config_key = "{}_{}".format(dataset_name, model_name)
    if config_key in self.CONFIGS:
      return self.CONFIGS[config_key]
    else:
      raise ValueError("No matching config {}.".format(config_key))

  @property
  def CONFIGS(self):
    return self._CONFIGS
