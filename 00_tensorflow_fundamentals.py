""" Fundamental concepts of tensors
1- Introduction to tensors
2- Getting informations from tensors
3- Manipulating tensors
4- Tensors and numpy
5- Using @tf.function (speeding python functions)
6- using GPUs with Tensors (TPU)
7- Exercises

-----Definitions- 1-----
Scalar: a single number (0 dimensional tensor)
Vector: a number with direction (1 dimensional tensor)
Matrix: a two dimensional array of numbers
Tensor: an  dimensional array of numbers

-----Definitions- 2-----
Sometimes we must have changeable and unchangeable tensors
tf.constant(): unchangeable tensor
tf.Variable(): changeable tensor
"""
import numpy as np
from random import seed
import tensorflow as tf
print(tf.__version__)

# ---------------Creating tensors with TF.CONSTANT()---------------
# Create tensors
scalar = tf.constant(7)
print(scalar)
# Check the number of dimensions of a tensor
print(scalar.ndim)

# Create a vector
vector = tf.constant([10, 10])
print(vector)
# Check the number of dimensions of a vector
print(vector.ndim)

# Create a matrix (multidemension vectors)
matrix = tf.constant([[10, 7], [7, 10]])
print(matrix)
# Check the number of dimensions of a matrix
print(matrix.ndim)
# Create another matrix, specifying the datatype
another_matrix = tf.constant([[10., 7.], [4., 2.], [5., 6.]], dtype=tf.float16)
print(another_matrix)
# Check the number of dimensions
print(another_matrix.ndim)

# Creating a tensor
tensor = tf.constant([[[1, 2, 3, ],
                       [4, 5, 6]],
                      [[7, 8, 9],
                      [10, 11, 12]],
                     [[13, 14, 15],
                      [16, 17, 18]]])
print(tensor)
# Checking the number of dimensions
print(tensor.ndim)

# ---------------Creating tensors with TF.VARIABLE()---------------
# Creating the above tensors with tf.variable()
changeable_tensor = tf.Variable([10, 7])
unchangeable_tensor = tf.constant([10, 7])
print(changeable_tensor, unchangeable_tensor)

# Changing changeable tensor
changeable_tensor[0].assign(7)
print(changeable_tensor)

# ---------------Creating random tensors---------------
# Initialize weights for the models
# Set pseudo random
random_1 = tf.random.Generator.from_seed(7)
random_1 = random_1.normal(shape=(3, 2))
print(random_1)
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2))
print(random_2)


# ---------------Shuffle the order of elements in a tensor---------------
# Does not bias the learning of the model using ordered data
# Global seed
tf.random.set_seed(42)
not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [2, 5]])
# Local seed
shuffle_tensor = tf.random.shuffle(not_shuffled, seed=42)
print(not_shuffled)
print(shuffle_tensor)

# ---------------Other ways to create tensors---------------
# Create a tensor of all ones
one_tensor = tf.ones([10, 7])
print(one_tensor)
# Create a tensor of all zeros
zero_tensor = tf.zeros([10, 7])
print(zero_tensor)

# ---------------Turn numpy arrays into tensors---------------
# The main difference between np arrays and tf tensors is that tensors can be run on a GPU much faster
# Create a numoy array between 1 and 25
numpy_A = np.arange(1, 25, dtype=np.int32)
print(numpy_A)
# Turn into a tensor
tensor_A = tf.constant(numpy_A)
print(tensor_A)
# Changing the shape
# The shape multiplication must be equal to the original shape
tensor_B = tf.constant(numpy_A, shape=(2, 3, 4))
print(tensor_B)
print(tensor_B.ndim)

# ---------------Getting informations from tensor---------------
