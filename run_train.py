from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from tqdm import tqdm
import tensorflow as tf

from src.data.data_handler import get_dataset
from src.models.model_handler import get_model
from src.configs.configs import Configs
from src.utils.dashboard_logger import DashboardLogger
from src.eval.evaluator import Evaluator

flags = tf.flags
flags.DEFINE_string("model", "siamese", "Model name")
# flags.DEFINE_string("model", "mAP", "Model name")
flags.DEFINE_string("dataset", "omniglot", "Dataset name")
# flags.DEFINE_string("dataset", "mini_imagenet", "Dataset name")

FLAGS = tf.flags.FLAGS


def _get_model(config):
  model = get_model(config)
  return model


def _get_datasets(dataset_name, config):
  train_dataset = get_dataset(dataset_name, config, "train")
  val_dataset = get_dataset(dataset_name, config, "val")
  test_dataset = get_dataset(dataset_name, config, "test")
  return train_dataset, val_dataset, test_dataset


def evaluate_siamese(sess, model, dataset, num_batches=100):
  summed_accs = 0
  for neval in tqdm(
      range(num_batches), desc="Computing validation siamese accuracy"):
    batch = dataset.next_batch()
    this_loss, this_acc = sess.run(
        [model.loss, model.siamese_accuracy],
        feed_dict={
            model.x: batch["imgs"],
            model.y: batch["pair_labels"],
            model.first_inds: batch["inds_A"],
            model.second_inds: batch["inds_B"]
        })
    summed_accs += this_acc

  return summed_accs / float(num_batches)


def train(sess, config, start_uidx, model, train_dataset, val_dataset,
          train_evaluator, val_evaluator, dashboard_logger, saver):

  best_val_acc = 0
  best_val_mAP = 0
  for uidx in tqdm(
      range(start_uidx, config.niters),
      desc="Training model {}".format(config.name)):

    just_reloaded = False
    if config.reload and uidx == start_uidx:
      just_reloaded = True
    train_batch = train_dataset.next_batch()

    # Save a model checkpoint
    if config.save_model and uidx % config.save_freq == 0 and uidx > start_uidx:
      if not os.path.isdir(config.saveloc):
        os.makedirs(config.saveloc)
      print("Saving model at {}".format(config.saveloc))
      saver.save(
          sess, os.path.join(config.saveloc, config.name), global_step=uidx)

    # Compute the validation performance
    if uidx % config.validation_freq == 0:
      if config.model_type == "siamese":
        val_acc = evaluate_siamese(sess, model, val_dataset,
                                   config.neval_batches)
        if val_acc > best_val_acc:
          best_val_acc = val_acc
        print(
            "Update {}, validation accuracy: {}, best validation accuracy so far {}".
            format(uidx, val_acc, best_val_acc))
      elif config.model_type == "mAP":
        mAPs = []
        for neval in tqdm(
            range(config.neval_batches), desc="Computing validation mAP"):
          val_batch = val_dataset.next_batch()
          mAPs.append(val_evaluator.eval_mAP(val_batch))
        val_mAP = sum(mAPs) / float(config.neval_batches)
        if val_mAP > best_val_mAP:
          best_val_mAP = val_mAP
        print("Update {}, validation mAP: {}, best validation mAP so far {}".
              format(uidx, val_mAP, best_val_mAP))

    # Compute mAP performance of siamese on train/validation sets
    if config.model_type == "siamese" and config.compute_mAP and uidx % config.compute_mAP_freq == 0:
      mAPs = []
      train_mAP = train_evaluator.eval_mAP(train_batch)
      for neval in tqdm(
          range(config.neval_batches), desc="Computing validation mAP"):
        val_batch = val_dataset.next_batch()
        mAPs.append(val_evaluator.eval_mAP(val_batch))
      val_mAP = sum(mAPs) / float(config.neval_batches)
      print("Update {}, train mAP: {}, validation mAP: {}".format(
          uidx, train_mAP, val_mAP))

    # Compute few-shot learning performance
    if config.compute_fewshot and uidx % config.fewshot_test_freq == 0:
      results = []
      for metric in config.few_shot_metrics:
        if metric["type"] == "classif":
          result, _ = val_evaluator.eval_fewshot_classif(
              metric["K"], metric["N"])
          results.append(result)
        elif metric["type"] == "retrieval":
          if not metric["K"] == 1:
            raise ValueError("Only 1-shot retrieval supported currently.")
          result, _ = val_evaluator.eval_oneshot_retrieval(metric["N"], 10)
          results.append(result)

    # Potentially adapt learning rate according to specified schedule
    if config.ada_learning_rate and uidx >= config.start_decr_lr and uidx % config.freq_decr_lr == 0 and not just_reloaded:
      current_lr = sess.run(model.lr)
      new_lr = current_lr * config.mult_lr_value
      if new_lr >= config.smallest_lr:
        sess.run(model.assign_lr, feed_dict={model.new_lr: new_lr})
        updated_lr = sess.run(model.lr)
        note = "Updated lr from {} to {} in uidx {}\n".format(
            current_lr, updated_lr, uidx)
      else:
        note = "Reached smallest lr value {}, omitting learning rate decrease.\n".format(
            config.smallest_lr)
      print(note)
      dashboard_logger.take_note(note)

    # Perform a training step
    if config.model_type == "siamese":
      train_loss, train_acc, _ = sess.run(
          [model.loss, model.siamese_accuracy, model.train_step],
          feed_dict={
              model.x: train_batch["imgs"],
              model.y: train_batch["pair_labels"],
              model.first_inds: train_batch["inds_A"],
              model.second_inds: train_batch["inds_B"]
          })
    elif config.model_type == "mAP":
      num_pos, num_neg, pos_inds, neg_inds = model.get_positive_negative_splits(
          train_batch)
      Y_aug = model.perform_loss_augmented_inference(sess, train_batch)
      _feed_dict = {
          model.x: train_batch["imgs"],
          model.n_queries_to_parse: model.config.batch_size,
          model.num_pos: num_pos,
          model.num_neg: num_neg,
          model.pos_inds: pos_inds,
          model.neg_inds: neg_inds,
          model.Y_aug: Y_aug
      }
      train_loss, _, score_std, score_aug = sess.run(
          [
              model.loss, model.train_step, model.mAP_score_std,
              model.mAP_score_aug
          ],
          feed_dict=_feed_dict)
      train_mAP = train_evaluator.eval_mAP(train_batch)

    # Update deep dashboard
    if uidx % config.update_dashboard_freq == 0 and not just_reloaded:
      if config.model_type == "siamese":
        dashboard_logger.write_acc(uidx, train_acc, val_acc)
      elif config.model_type == "mAP":
        dashboard_logger.write_F_score(uidx, score_std, score_aug)
      if config.model_type == "mAP" or config.compute_mAP:
        dashboard_logger.write_mAP(uidx, train_mAP, val_mAP)
      if config.compute_fewshot:
        for result, metric in zip(results, config.few_shot_metrics):
          if metric["type"] == "classif":
            dashboard_logger.write_Kshot_Nway_classif(uidx, metric["K"],
                                                      metric["N"], result)
          elif metric["type"] == "retrieval":
            dashboard_logger.write_oneshot_Nway_retrieval(
                uidx, metric["N"], result)

    # Display training progress
    if uidx % config.display_step == 0:
      if config.model_type == "siamese":
        print(
            "Update {}, train accuracy: {}, best validation accuracy so far: {}".
            format(uidx, train_acc, best_val_acc))
      elif config.model_type == "mAP":
        print("Update {}, train mAP: {}, best validation mAP so far: {}".format(
            uidx, train_mAP, best_val_mAP))
      print("Train loss {}".format(train_loss))


def main():
  configs = Configs()
  config = configs.get_config(FLAGS.dataset, FLAGS.model)

  train_dataset, val_dataset, test_dataset = _get_datasets(
      FLAGS.dataset, config)

  model = _get_model(config)
  dashboard_logger = DashboardLogger(config)
  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:
    if config.reload:
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
        print("Continuing from update uidx: " + str(uidx))
      else:
        raise ValueError(
            "No checkpoint to restore from in {}".format(config.saveloc))

      # If using an adaptive learning rate schedule,
      # resume from the appropriate point
      if config.ada_learning_rate:
        current_lr = config.lr
        for uidx_ in range(uidx + 1):
          if uidx_ >= config.start_decr_lr and uidx_ % config.freq_decr_lr == 0:
            new_lr = current_lr * config.mult_lr_value
            if new_lr >= config.smallest_lr:
              current_lr = new_lr
        config.lr = current_lr
        note = "Reloaded from uidx {} and using lr {}\n".format(uidx, config.lr)
        print(note)
        dashboard_logger.take_note(note)
    else:
      uidx = 0
      sess.run(tf.global_variables_initializer())

    # Create Evaluator objects
    train_evaluator = Evaluator(config, model, train_dataset, sess)
    val_evaluator = Evaluator(config, model, val_dataset, sess)

    train(sess, config, uidx, model, train_dataset, val_dataset,
          train_evaluator, val_evaluator, dashboard_logger, saver)


if __name__ == "__main__":
  main()
