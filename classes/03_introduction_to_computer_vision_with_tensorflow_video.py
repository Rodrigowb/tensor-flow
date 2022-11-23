"""
Introduction to Convolution Neural Networks and Compuer Vision with Tensorflow

-----Steps into modeling CNN-----
1- Load the images
2- Preprocess the images
3- Build a CNN to find patterns in the image
4- Compile the CNN
5- Fit the CNN to our training data

-----Batches-----
Control memory in the computer.

-----Basic architecture-----
1. Convolutional layer (+ Relu activation)
2. Pooling layer
...

-----Layers-----
1. In the first layers, we might need to tell the model the input shape

-----Hyperparameters-----
1. Filters: higher values lead to more complex models
2. Kerne size (filter size): determines the shape of the filters (lower values learns smaller features)
3. Padding: pads the target tensor with zeroes os leaves as it is (same or valid)
4. Strides: the numbers of steps (pixels) a filter takes across an image at a time (1 or 2)

-----Adjust the model parameters-----
1. Create a baseline
2. Beat the baseline
3. Reduce overfitting

-----Ways to induce overfitting-----
1. Increase the numbers of conv layers
2. Increase the numbers of conv filters
3. Add another dense layer to the output of our flatten layer

-----Ways to reduce overfitting-----
1. Add data augmentation
2. Add regularization layers (such as MaxPool2D)
3. Add more data 
4. Simplify the model (remove layers and decrese the number of hidden units ) 
5. Use transfer learning

-----Max Pooling Layers-----
Condense the outputs of the Conv2D layers, by keeping only the most important, so the model can learn better

-----Data augmentation-----
Processing of altering our training data, leading to more diversity and making the model learn better.
Adjust the rotation, flipping it, cropping it...

-----Improvements after baseline-----
1. Increase the number of model layers
2. Increase the number of filters in each convolutional layer (from 10 to 32, even 64)
3. Train for longer
4. Find an ideal learning rate
5. Get more data
6. Use transfer learning to leverage what another image model has learn and adjust to our user case

-----Binary vs. multiclass-----
1. Output layer:
- Binary: Sigmoid
- Multiclass: Softmax
2. Loss function:
- Binary: BinaryCrossEntropy()
- Multiclass: CategoricalCrossentropy()

"""

# from services import MachineLearningMetrics as metrics
import sys
sys.path.insert(
    0, '/Users/rodrigow/Desktop/Personal-projects/tensor-flow-course/services')
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from unittest import result
from pickletools import optimize
import os
from multiprocessing import pool
import sys
from services import MachineLearningMetrics as metrics


def directory_details(dir_path):
    """Pass a directory and return the numbers of files in it"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f'There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}')


def plot_loss_curve(history):
    """Return separate loss curves for training and validation metrics."""

    # Get the history metrics
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(history.history["loss"]))

    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot graphs
    plt.show()


def view_random_image(target_folder):
    """Get a random image from a especific folder and show it"""
    random_image = random.sample(os.listdir(target_folder), 1)
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.axis("off")
    print(f"Image shape: {img.shape}")
    print(img.dtype)
    plt.show()
    return tf.constant(img)

def load_and_pred_image(filename, img_shape=224):
  """Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, colour_chanels)"""
  # Read the image
  img = tf.io.read_file(filename)
  # Decode the readed file into a tensor
  img = tf.image.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, size=[img_shape, img_shape])
  # Rescale the image (pixels betweeen 0 and 1)
  img = img/255.
  return img

def pred_and_plot(model, filename, class_names):
  """Imports an image, makes a prediction and plo the image with the label as the title"""

  # Import the target image and preprocess it
  img = load_and_pred_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted classes
  # Multiclass
  if len(pred[0]) > 1:
    pred_class = class_names[tf.argmax(pred[0])]
  # Binary
  else:
    pred_class = class_names[int(tf.round(pred))]

  # Plot the image
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)
  plt.show()


def first_cnn(run=True, load=False, save=False):
    # Set random seed
    tf.random.set_seed(42)

    # Preprocess data (divide all pixel values to 255 for normalization)
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Data augmentation
    train_datagen_augmented = ImageDataGenerator(
        rescale=1./255,
        rotation_range=0.2,
        # shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.3,
        horizontal_flip=True)

    # Set up paths for the directories
    train_dir = "../services/images/pizza_steak/train"
    test_dir = "../services/images/pizza_steak/test"

    # Import the data and transform them into batches
    train_data = train_datagen.flow_from_directory(
        directory=train_dir, batch_size=32, target_size=(224, 224), class_mode="binary", seed=42)

    valid_data = test_datagen.flow_from_directory(
        directory=test_dir, batch_size=32, target_size=(224, 224), class_mode="binary", seed=42)

    # Data augmentation
    train_data_augmented = train_datagen_augmented.flow_from_directory(
        directory=train_dir, batch_size=32, target_size=(224, 224), class_mode="binary", seed=42)

    # Run model
    if run:
      # Build a CNN model
      model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(filters=10,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding="valid",
                                activation="relu",
                                input_shape=(224, 224, 3)),
          tf.keras.layers.MaxPool2D(pool_size=2),
          tf.keras.layers.Conv2D(10, 3, activation="relu"),
          tf.keras.layers.MaxPool2D(),
          tf.keras.layers.Conv2D(10, 3, activation="relu"),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(1, activation="sigmoid")
      ])

      # Compile the CNN
      model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

      # Fit the CNN model
      history = model.fit(train_data_augmented, epochs=20, steps_per_epoch=len(
          train_data_augmented), validation_data=valid_data, validation_steps=len(valid_data))

      # Plot loss curve (check if the model if overfitted)
      plot_loss_curve(history=history)

      # Make predictions
      pred_and_plot(model, '../services/images/pizza_steak/pred/pizza-pred.jpeg', ['pizza', 'steak'])

    # Save the model
    if save:
      # Saving and loading our model(can be lounched as an API)
      model.save("saved_trained_binary_food_classification")

    # Load the model
    if load:
      loaded_model = tf.keras.models.load_model("saved_trained_binary_food_classification")
      loaded_model.evaluate(valid_data)
    

if __name__ == "__main__":
    # directory_details("../services/images/pizza_steak")
    # image = view_random_image("../services/images/pizza_steak/test/steak")
    first_cnn(run=False, load=True, save=False)
