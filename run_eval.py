import os
import tensorflow as tf

from src.data.data_handler import get_dataset
from src.models.model_handler import get_model
from src.configs.configs import Configs
from src.eval.evaluator import Evaluator

flags = tf.flags
# flags.DEFINE_string("model", "siamese", "Model name")
flags.DEFINE_string("model", "mAP", "Model name")
# flags.DEFINE_string("dataset", "omniglot", "Dataset name")
flags.DEFINE_string("dataset", "mini_imagenet", "Dataset name")

FLAGS = tf.flags.FLAGS

OUTDIR = os.path.join("results", FLAGS.dataset)

if __name__ == "__main__":
  configs = Configs()
  config = configs.get_config(FLAGS.dataset, FLAGS.model)

  test_dataset = get_dataset(FLAGS.dataset, config, "test")
  model = get_model(config)
  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:

    ckpt = tf.train.latest_checkpoint(config.saveloc)
    if ckpt:
      saver.restore(sess, ckpt)
      print("Restored weights from {}".format(config.saveloc))

      # Find out the uidx that we are restoring from
      with open(os.path.join(config.saveloc, "checkpoint"), "r") as f:
        lines = f.readlines()
      model_checkpoint_line = lines[0].strip()
      dash_ind = model_checkpoint_line.rfind('-')
      uidx = int(model_checkpoint_line[dash_ind + 1:-1])
      print("Restored from update {}".format(uidx))
    else:
      raise ValueError(
          "No checkpoint to restore from in {}".format(config.saveloc))

    # Create an Evaluator object
    test_evaluator = Evaluator(config, model, test_dataset, sess)

    # Perform evaluation
    if config.dataset == "omniglot":
      oneshot_5way_mAP, _ = test_evaluator.eval_oneshot_retrieval(
        5, 10, num_samples=1000)
      oneshot_20way_mAP, _ = test_evaluator.eval_oneshot_retrieval(
        20, 10, num_samples=1000)
      oneshot_5way_acc, _ = test_evaluator.eval_fewshot_classif(
          1, 5, num_samples=1000)
      oneshot_20way_acc, _ = test_evaluator.eval_fewshot_classif(
          1, 20, num_samples=1000)
      if not FLAGS.model == "siamese":
        fiveshot_5way_acc, _ = test_evaluator.eval_fewshot_classif(
          5, 5, num_samples=1000)
        fiveshot_20way_acc, _ = test_evaluator.eval_fewshot_classif(
          5, 20, num_samples=1000)

    elif config.dataset == "mini_imagenet":
      oneshot_5way_acc_mean, oneshot_5way_acc_pm = test_evaluator.eval_fewshot_classif(
          1, 5, num_samples=600)
      if not FLAGS.model == "siamese":
        fiveshot_5way_acc_mean, fiveshot_5way_acc_pm = test_evaluator.eval_fewshot_classif(
          5, 5, num_samples=600)
      oneshot_5way_mAP_mean, oneshot_5way_mAP_pm = test_evaluator.eval_oneshot_retrieval(
        5, 10, num_samples=600)
      oneshot_20way_mAP_mean, oneshot_20way_mAP_pm = test_evaluator.eval_oneshot_retrieval(
        20, 10, num_samples=600)

    # Save results to file
    if not os.path.isdir(OUTDIR):
      os.makedirs(OUTDIR)
    with open(os.path.join(OUTDIR, config.name + ".txt"), "a") as f:
      if config.dataset == "omniglot":
        f.write(
            "Results from model {} at update {}:\n".format(config.name, uidx))
        f.write("1-shot 5-way acc {}\n".format(oneshot_5way_acc))
        f.write("1-shot 20-way acc {}\n".format(oneshot_20way_acc))
        if not FLAGS.model == "siamese":
          f.write("5-shot 5-way acc {}\n".format(fiveshot_5way_acc))
          f.write("5-shot 20-way acc {}\n".format(fiveshot_20way_acc))
        f.write("1-shot 5-way mAP {}\n".format(oneshot_5way_mAP))
        f.write("1-shot 20-way mAP {}\n".format(oneshot_20way_mAP))
      elif config.dataset == "mini_imagenet":
        f.write(
            "Results from model {} at update {}:\n".format(config.name, uidx))
        f.write("1-shot 5-way acc {} plus/minus {}\n".format(
            oneshot_5way_acc_mean, oneshot_5way_acc_pm))
        if not FLAGS.model == "siamese":
          f.write("5-shot 5-way acc {} plus/minus {}\n".format(
            fiveshot_5way_acc_mean, fiveshot_5way_acc_pm))
        f.write("1-shot 5-way mAP {} plus/minus {}\n".format(
            oneshot_5way_mAP_mean, oneshot_5way_mAP_pm))
        f.write("1-shot 20-way mAP {} plus/minus {}\n".format(
            oneshot_20way_mAP_mean, oneshot_20way_mAP_pm))
