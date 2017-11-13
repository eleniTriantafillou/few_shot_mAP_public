import os
from src.configs.generic_omniglot_config import GenericConfigOmniglot

SAVE_LOC = "saved_models/"
DASHBOARD_LOC = "/u/eleni/public_html/results/few_shot_mAP/"


class MeanAveragePrecisionConfigOmniglot(GenericConfigOmniglot):

  def __init__(self):
    # Copy the generic Omniglot options
    super(MeanAveragePrecisionConfigOmniglot, self).__init__()

    self.model_type = "mAP"

    # Learning rate value and schedule
    self.lr = 0.001
    self.ada_learning_rate = False
    self.start_decr_lr = 2000
    self.mult_lr_value = 0.5
    self.freq_decr_lr = 2000
    self.smallest_lr = 0.0001

    # Optimization
    self.optimizer = "ADAM"
    self.epsilon = 1
    self.alpha = 10
    self.optimization_framework = "DLM"  # Direct Loss Minimization
    # self.optimization_framework = "SSVM"
    self.positive_update = True

    # Batch formation
    self.batch_size = 128
    self.nway = 16  # number of classes allowed in each batch

    self.reload = False

    self.name = "mAP_DLM_omniglot"
    # self.name = "mAP_SSVM_omniglot"
    
    # Metrics to plot throughout training
    self.few_shot_metrics = [{
        "K": 1,
        "N": 5,
        "type": "classif"
    }, {
        "K": 1,
        "N": 5,
         "type": "retrieval"
    }]

    # deep dashboard location
    self.dashboard_path = os.path.join(DASHBOARD_LOC, self.name)

    # where to save checkpoints of training models
    self.saveloc = os.path.join(os.path.join(SAVE_LOC, self.dataset), self.name)
