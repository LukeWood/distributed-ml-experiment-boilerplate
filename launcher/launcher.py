import tensorflow_cloud as tfc

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'train', [
    'train', 'eval'], 'Which mode to run the job in.')
flags.DEFINE_string('dataset', 'mnist', 'Tensorflow datset to use.')
flags.DEFINE_string('train_split', 'train',
                    'Dataset split to use for training.')
flags.DEFINE_integer('batch_size', 64, 'Batch size to use.')

flags.DEFINE_string('gcp_project', None, 'GCP project to run the training job', required=True)
flags.DEFINE_string('gcs_bucket', None,
                    'Google Cloud Storage bucket to save results to', required=True)

# Config settings
flags.DEFINE_bool('silence_logs', True,
                  'Whether or not to silence Tensorflow logs.')


def convert_args_to_array(args):
    result = []
    for key in args:
        result.push(f'--{key}')
        result.push(args[key])
    return result


def main(argv):
    del argv

    args = {}
    args['dataset'] = FLAGS.dataset
    args['train_split'] = FLAGS.train_split
    args['mode'] = FLAGS.mode
    args['log_dir'] = f'gs://{FLAGS.gcs_bucket}/logs-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # keras interpolates {epoch} for us
    args['checkpoint_path'] = f'gs://{FLAGS.gcs_bucket}/{FLAGS.dataset}_model/' + '{epoch}'

    tfc.launch(
        entrypoint='main.py',
        chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
        worker_count=1,
        worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"],
        entry_point_args=convert_args_to_array(args)
    )


if __name__ == '__main__':
    app.run(main)
