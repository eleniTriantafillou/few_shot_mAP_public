import os
from src.configs.generic_mini_imagenet_config import GenericConfigMiniImageNet

SAVE_LOC = "saved_models/"
DASHBOARD_LOC = "/u/eleni/public_html/results/few_shot_mAP/"


class MeanAveragePrecisionConfigMiniImageNet(GenericConfigMiniImageNet):

  def __init__(self):
    # Copy the generic Omniglot options
    super(MeanAveragePrecisionConfigMiniImageNet, self).__init__()

    self.model_type = "mAP"

    # Learning rate value and schedule
    self.lr = 0.001
    self.ada_learning_rate = True
    self.start_decr_lr = 2000
    self.mult_lr_value = 0.5
    self.freq_decr_lr = 2000
    self.smallest_lr = 0.00001

    # Optimization
    self.optimizer = "ADAM"
    self.epsilon = 1
    self.alpha = 10
    self.optimization_framework = "DLM"  # Direct Loss Minimization
    # self.optimization_framework = "SSVM"
    self.positive_update = True

    # Batch formation
    # self.batch_size = 64
    self.batch_size = 128
    # self.batch_size = 16
    self.nway = 8  # number of classes allowed in each batch

    self.reload = False

    self.name = "mAP_DLM_miniImageNet"

    self.few_shot_metrics = [{
        "K": 1,
        "N": 5,
        "type": "classif"
    }, {
        "K": 5,
        "N": 5,
        "type": "classif"
    }, {
        "K": 1,
        "N": 5,
        "type": "retrieval"
    }, {
        "K": 1,
        "N": 20,
        "type": "retrieval"
    }]

    # deep dashboard location
    self.dashboard_path = os.path.join(DASHBOARD_LOC, self.name)

    # where to save checkpoints of training models
    self.saveloc = os.path.join(os.path.join(SAVE_LOC, self.dataset), self.name)
