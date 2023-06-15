"""
-----Transfer learning-----

Use pre build models to improve the performance of our models and adapt them into our use case.

-----Callbacks-----

Extra functionalities we can add to the models to be performed during or after training. Most popular:
1. Tensorboard: Tracking experiments.
2. ModelCheckpoint: Model checkpoint with callback.
3. EarlyStopping: Stopping a model from training (before it trains too long and overfits).

-----TensorFlow Hub-----
- Use pretreined models: https://tfhub.dev/

"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """Create a function to call TensorBoard callback."""
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f'Saving TensorBoard log files to: {log_dir}')
  return tensorboard_callback 

def first_transfer():

  tf.random.set_seed(42)

  BATCH_SIZE = 32
  TARGET_SIZE = (224, 224)

  train_datagen = ImageDataGenerator(rescale=1./255)
  test_datagen = ImageDataGenerator(rescale=1./255)

  train_dir = "../services/images/10_foo_classes_10_percent/train"
  test_dir = "../services/images/10_foo_classes_10_percent/test"

  train_data = train_datagen.flow_from_directory(directory=train_dir,
                                                  batch_size=BATCH_SIZE,
                                                  target_size=TARGET_SIZE,
                                                  class_mode="categorical",
                                                  seed=42)
  
  test_data = test_datagen.flow_from_directory(directory=test_dir,
                                                batch_size=BATCH_SIZE,
                                                target_size=TARGET_SIZE,
                                                class_mode="categorical",
                                                seed=42)

  
if __name__ == "__name__":
  first_transfer()