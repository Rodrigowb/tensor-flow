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
"""

from services import MachineLearningMetrics as metrics
from multiprocessing import pool
import os
from pickletools import optimize
from unittest import result
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import sys
sys.path.insert(0, '..')


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


def first_cnn():
    # Set random seed
    tf.random.set_seed(42)

    # Preprocess data (divide all pixel values to 255 for normalization)
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Data augmentation
    train_datagen_augmented = ImageDataGenerator(
        rescale=1./255,
        rotation_range=0.2,
        shear_range=0.2,
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
    history = model.fit(train_data_augmented, epochs=5, steps_per_epoch=len(
        train_data_augmented), validation_data=valid_data, validation_steps=len(valid_data))

    # Plot loss curve
    plot_loss_curve(history=history)

    # Find the ideal leraning rate
    metrics.plot_loss_vs_learning(history=history)


if __name__ == "__main__":
    # directory_details("../services/images/pizza_steak")
    # image = view_random_image("../services/images/pizza_steak/test/steak")
    first_cnn()
