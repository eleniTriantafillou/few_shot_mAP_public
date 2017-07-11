class GenericConfigMiniImageNet(object):
  """Contains options that are probably common to different models
  for training on miniImageNet."""

  def __init__(self):
    self.dataset = "mini_imagenet"
    self.batch_size = 64
    self.nway = 8
    self.height = 84
    self.width = 84
    self.channels = 3
    self.num_fewshot_samples = 100
    self.optimizer = "ADAM"
    self.niters = 30000
    self.compute_mAP = True
    self.compute_fewshot = True
    self.display_step = 20
    self.validation_freq = 500
    self.save_freq = 2000
    self.update_dashboard_freq = 100
    self.compute_mAP_freq = 500
    self.fewshot_test_freq = 1000
    self.save_model = True
    self.neval_batches = 100