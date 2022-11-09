"""Introduction to Convolution Neural Networks and Compuer Vision with Tensorflow"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf


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


if __name__ == "__main__":
    directory_details("../services/images/pizza_steak")
    image = view_random_image("../services/images/pizza_steak/test/steak")
    print(image)
