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

-----Steps to improve our model-----
Altering the steps to create a model
1- Creating a model:
a) Add more layers
b) Increase numbers of hidden layers (neurons)
c) Change activation function for each layer

2- Compiling the model:
a) Change the optimization function and learning rate of the optimization function

3- Fitting the model:
a) Set more epochs to the fitting (leave it training longer)
b) Get more data to the model train

-----Visualization for improving the model-----
1- The data
2- The model
3- The training
4- The predictions

-----The three sets-----
Fit and evaluate on different datasets
1- Training set: the model learns: 70-80% 
2- Validation set: the model gets tuned: 10-15%
3- Test set: the model gets evaluated to test what is has learned: 10-15%
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------Creating data to view and fit---------------
# # Creating features
# x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
# # Creating labels
# y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
# # Visualize it
# # plt.scatter(x, y)
# # plt.show()
# # Transform the arrays into tensors
# X = tf.constant(x)
# Y = tf.constant(y)
# print(X.shape)
# print(Y.shape)

# ---------------Steps in modeling with TF---------------


def FirstModel():
    # Creating features
    x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
    # Creating labels
    y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
    X = tf.constant(x)
    Y = tf.constant(y)

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

# ---------------Improving our model (rebuilding the model)---------------


def SecondModel():
    """
    Improving the FirstModel()
    1- Add epochs
    2- Adding hidden layers
    3- Increase the number of hidden units
    4- Change the activation function
    5- Change the optimization function
    6- Change the learning rate
    """
    # Creating features
    x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
    # Creating labels
    y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
    X = tf.constant(x)
    Y = tf.constant(y)

    # 1- Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation=None),
        # tf.keras.layers.Dense(100, activation="relu"),
        # tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(1)

    ])

    # 2- Compiling the model
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  metrics=["mae"])

    # 3- Fit the model
    model.fit(tf.expand_dims(X, axis=-1), Y, epochs=100)

    # 4- Try to predict using the builded model
    prediction = model.predict([17.0])
    print(prediction)

    # !!!Producing overfitting: the metrics in the training data is not the key parameters to evaluate a model!!!


# SecondModel()
# FirstModel()

# ---------------Evaluating a model---------------

def ThirdModel():
    # Making a bigger dataset
    X = tf.range(-100, 100, 4)
    print(X)
    # Make labels for the dataset
    Y = X + 10
    print(Y)

    # 1- Visualize the data
    plt.plot(X, Y)
    # plt.show()


ThirdModel()