import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

# User modules
from train import train

FLAGS = flags.FLAGS

# High level args
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Which mode to run the job in.')

# Dataset tuning
flags.DEFINE_string('dataset', 'mnist', 'Tensorflow datset to use.')
flags.DEFINE_string('train_split', 'train', 'Dataset split to use for training.')
flags.DEFINE_integer('batch_size', 64, 'Batch size to use.')

# Exporting/metric logging settings
flags.DEFINE_string('gcs_bucket', None, 'Google cloud storage bucket to save results to.')

# Config settings
flags.DEFINE_bool('silence_logs', True, 'Whether or not to silence Tensorflow logs.')

def main(argv):
    del argv
    logging.info("Current distribution strategy: %s", tf.distribute.get_strategy())
    if FLAGS.mode == 'train':
            train(FLAGS.dataset, FLAGS.train_split, FLAGS.batch_size)


if __name__ == '__main__':
    app.run(main)

