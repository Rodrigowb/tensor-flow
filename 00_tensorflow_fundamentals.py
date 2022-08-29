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
import tensorflow_probability as tfp
from statistics import multimode
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
# Create a rank 4 tensor
rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5])
print(rank_4_tensor)
# Shape: tensor.shape
print(rank_4_tensor.shape)
# Rank: tensor.ndim
print(rank_4_tensor.ndim)
# Axis (or dimension): tensor[0], tensor[:, 1]...
print(rank_4_tensor[0])
# Size: tf.size(tensor)
print(tf.size(rank_4_tensor).numpy())

# ---------------Indexing tensors---------------
# Tensors can be indexed just like Python lists
# Getting the first two elements
print(rank_4_tensor[:2, :2, :2, :2])
#  Getting the first element from each dimension from each index index excepto from the final one
print(rank_4_tensor[:1, :1, :1, :])

# ---------------Changing tensor dimension---------------
rank_2_tensor = tf.zeros(shape=(2, 3))
print(rank_2_tensor)
# Getting the last item of each row
print(rank_2_tensor[:, -1])
# Adding extra dimensios (rank 2 --> rank 3)
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
print(rank_2_tensor)
print(rank_3_tensor)
# Alternative to tf.newaxis (-1 means extend the final axis)
rank_3_tensor_expand = tf.expand_dims(rank_2_tensor, axis=-1)
print(rank_3_tensor_expand)

# ---------------Manipulating tensors (basic tensors operations)---------------
# Adding values from the tensors (+)
tensor = tf.constant([[10, 7], [3, 4]])
print(tensor + 10)
print(tf.add(tensor, 10))
# Multiplicating values from the tensors (*)
print(tensor * 2)
print(tf.multiply(tensor, 2))
# Substracting values from the tensors (-)
print(tensor - 6)
print(tf.subtract(tensor, 6))
# Dividing values from the tensors (/)
print(tensor/2)
print(tf.divide(tensor, 2))

# ---------------Matrix multiplication P1---------------
# Dot product: matrixmultiplication.xyz
# Most of the common tensor operations in ML
# TF way
multiplied_tensor = tf.matmul(tensor, tensor)
print(multiplied_tensor)
# Python way
multiplied_tensor_pway = tensor @ tensor
print(multiplied_tensor_pway)

# ---------------Matrix multiplication P2---------------
# Rules for matrix multiplication (transpose or reshape to match)
# 1- The inner dimension must match
# 2- The resulting matrix has the shape of the outer dimensions
X = tf.constant([[1, 2], [1, 2]])
Y = tf.constant([[1, 2], [1, 2], [1, 2]])
# This will not work, because the inner dimensions are not equal
# X_mult_Y = tf.matmul(X, Y)
# Reshape the matrix
Y_reshaped = tf.reshape(Y, shape=(2, 3))
print(Y_reshaped)
X_mult_Y = tf.matmul(X, Y_reshaped)
# Worked because now the inner dimensions are equal (2x2 and 2x3)
print(X_mult_Y)

# ---------------Matrix multiplication P3---------------
# We can miltiplicate matrix using:
# 1- tf.matmul()
# 2- tf.tensordot() (specifies the indexes)
# Using matmul
Y_transposed = tf.transpose(Y)
X_mult_Y_transposed = tf.matmul(X, Y_transposed)
print(X_mult_Y_transposed)
#  Using ternsordot
mult_dot = tf.tensordot(X, Y_transposed, 1)
print(mult_dot)

# ---------------Changing the datatype of a tensor---------------
# Most datatype will be int32
B = tf.constant([[1.7, 7.4]])
print(B.dtype)
C = tf.constant([[7, 4]])
print(C.dtype)
# Change from float32 to float16 (reduce precision)
D = tf.cast(B, dtype=tf.float16)
print(D.dtype)

# ---------------Aggregating tensors---------------
# Condensing tensors from multiple values down to a smaller amount of values
# Get the absolute values (transform all values to positive)
E = tf.constant([[-10, -7], [5, 7], [9, 7], [50, 30]])
print(tf.abs(E))
# Forms of aggregation
F = tf.constant(np.random.randint(0, 500, size=50))
print(F)
# 1- Minimum
F_min = tf.reduce_min(F)
print(F_min)
# 2- Maximum
F_max = tf.reduce_max(F)
print(F_max)
# 3- Mean
F_mean = tf.reduce_mean(F)
print(F_mean)
# 4- Sum
F_sum = tf.reduce_sum(F)
print(F_sum)
# 5- Variance
F_variance = tfp.stats.variance(F)
print(F_variance)
# 6- Std
F_std = tf.math.reduce_std(tf.cast(E, dtype=tf.float32))
print(F_std)

# ---------------Finding the positional maximun and minimun---------------
