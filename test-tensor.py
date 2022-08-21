""" Fundamental concepts of tensors
1- Introduction to tensors
2- Getting informations from tensors
3- Manipulating tensors
4- Tensors and numpy
5- Using @tf.function (speeding python functions)
6- using GPUs with Tensors (TPU)
7- Exercises
"""
import tensorflow as tf
print('Hello Word!')
print(tf.__version__)

# Create tensors
scalar = tf.constant(7)
print(scalar)
