import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers
from absl import logging

from model import create_model

def preprocess(x, label, num_classes=10):
    return tf.cast(x, tf.float32) / 255.0, tf.cast(tf.one_hot(label, num_classes), tf.float32)

def train(dataset: str, train_split: str, batch_size: int):
  logging.info("Kicking off training") 
  dataset = tfds.load(name=dataset, split=train_split, as_supervised=True, try_gcs=True)
  dataset = dataset.batch(batch_size).map(preprocess) 

  model = create_model()
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  model.build(dataset.element_spec[0].shape)
  model.summary()
  model.fit(dataset, epochs=3)
