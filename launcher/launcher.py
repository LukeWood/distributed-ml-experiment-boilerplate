import tensorflow_cloud as tfc

from absl import app
from absl import flags

FLAGS = flags.flags


def main(argv):
    del argv
    
    tfc.launch(
       entrypoint='main.py' ,
       chief_config=tfc.COMMON_MACHINE_CONFIGS["CPU"],
       worker_count=1,
       worker_config=tfc.COMMON_MACHINE_CONFIGS["TPU"])
    )

if __name__ == '__main__':
    absl.run(main)
