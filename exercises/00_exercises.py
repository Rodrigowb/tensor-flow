import tensorflow as tf
import numpy as np

# Create a vector, scalar, matrix and tensor
vector = tf.constant(10)
scalar = tf.constant([10, 5])
matrix = tf.constant([[10, 5], [20, 10]])
tensor = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print(tensor)

# Finding the shape. rank and size of the tensor created
print(tensor.shape)
print(tensor.ndim)
print(tf.size(tensor))

# Create two tensors with random values between 0 and 1 with shape [5, 300]

tensor_1 = tf.random.uniform(shape=[5, 300], minval=0, maxval=1)
print(tensor_1)
print(tensor_1.shape)
tensor_2 = tf.random.uniform(shape=[5, 300], minval=0, maxval=1)
print(tensor_2)
print(tensor_2.shape)