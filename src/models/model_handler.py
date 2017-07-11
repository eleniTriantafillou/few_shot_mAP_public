from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from src.models.siamese import SiameseModel
from src.models.mAP_model import MeanAveragePrecisionModel


def get_model(config, reuse=False):
  if config.model_type == "siamese":
    return SiameseModel(config, reuse=reuse)
  elif config.model_type == "mAP":
    return MeanAveragePrecisionModel(config, reuse=reuse)
  else:
    raise ValueError("Unknown model type. \"{}\"".format(config.model_type))
