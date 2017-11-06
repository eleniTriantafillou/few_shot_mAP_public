import os
from src.configs.generic_omniglot_config import GenericConfigOmniglot

SAVE_LOC = "saved_models/"
DASHBOARD_LOC = "/u/eleni/public_html/results/few_shot_mAP/"


class SiameseConfigOmniglot(GenericConfigOmniglot):

  def __init__(self):
    # Copy the generic Omniglot options
    super(SiameseConfigOmniglot, self).__init__()

    # Siamese network
    self.model_type = "siamese"
    self.loss_function = "cross_entropy"
    self.join_branches = "abs_diff"

    # Learning rate value and schedule
    self.lr = 0.1
    self.ada_learning_rate = False
    self.start_decr_lr = 2000
    self.mult_lr_by = 0.5
    self.freq_decr_lr = 2000
    self.smallest_lr = 0.0001

    # Optimization
    self.optimizer = "ADAM"

    # Batch formation
    self.batch_size = 64
    self.nway = 8  # number of classes allowed in each batch

    self.reload = False

    self.name = "siamese_omniglot"

    self.few_shot_metrics = [{
        "K": 1,
        "N": 5,
        "type": "classif"
    }, {
        "K": 1,
        "N": 20,
        "type": "classif"
    }, {
        "K": 1,
        "N": 20,
        "type": "retrieval"
    }]

    # deep dashboard location
    self.dashboard_path = os.path.join(DASHBOARD_LOC, self.name)

    # where to save checkpoints of training models
    self.saveloc = os.path.join(os.path.join(SAVE_LOC, self.dataset), self.name)
