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
"""

from multiprocessing import pool
import os
from pickletools import optimize
from unittest import result
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def directory_details(dir_path):
    """Pass a directory and return the numbers of files in it"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f'There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}')


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

    # Preprocess data
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Set up paths for the directories
    train_dir = "../services/images/pizza_steak/train"
    test_dir = "../services/images/pizza_steak/test"

    # Import the data and transform them into batches
    train_data = train_datagen.flow_from_directory(
        directory=train_dir, batch_size=32, target_size=(224, 224), class_mode="binary", seed=42)

    valid_data = test_datagen.flow_from_directory(
        directory=test_dir, batch_size=32, target_size=(224, 224), class_mode="binary", seed=42)

    # Build a CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=10, kernel_size=3,
                               activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, padding="valid"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Compile the CNN
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    # Fit the CNN model
    history = model.fit(train_data, epochs=5, steps_per_epoch=len(
        train_data), validation_data=valid_data, validation_steps=len(valid_data))


if __name__ == "__main__":
    # directory_details("../services/images/pizza_steak")
    # image = view_random_image("../services/images/pizza_steak/test/steak")
    first_cnn()
