"""
-----Regression with Neural Network-----
Predicting a numerical variable based o some other combinations of variables (Predicting a number).

-----Architecture of a regression model-----
1- Input layer shape
2- Hidden layers
3- Neurons per hidden layers
4- Output layer shape
5- Hidden activation
6- Output activation
7- Loss function
8- Optimizer

-----Steps in modeling with TensorFlow-----
1- Creating a model: define the input, hidden and output layers.
2- Compiling a model: define the loss function (tells how wrong the model is) and the optimizer 
how to improve the patterns) and evaluating metrics (what we can use to interpret the performance of our model).
3- Fitting a model: letting the model try to find patterns between x and y (features and labels).
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------Creating data to view and fit---------------
# Creating features
x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
# Creating labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
# Visualize it
# plt.scatter(x, y)
# plt.show()
# Transform the arrays into tensors
X = tf.constant(x)
Y = tf.constant(y)
print(X.shape)
print(Y.shape)

# ---------------Steps in modeling with TF---------------
# Set random seed
tf.random.set_seed(42)

# 1. Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# 2. Compile the error (error between labels and predictions)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# 3. Fit the model (after it, you have a trained model)
model.fit(tf.expand_dims(X, axis=-1), Y, epochs=1000)

# 4. Try and make a prediction using the model
prediction = model.predict([17.0])
print(prediction)
