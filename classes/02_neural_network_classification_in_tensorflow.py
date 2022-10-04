"""
-----Typical architecture of a classification model-----
1- Input layer shape: numbers of features
2- Hidden layers: numbers of features recommended
3- Neurons per hidden layer: 10 to 100 (decendent)
4- Output layers shape: 1 per class
5- Hidden activation: ususally Relu
6- Output activation: Sigmoid (binary), Softmax (multiclass)
7- Loss function: Cross entropy
8- Optimizer: SGD or Adam

-----Types of classification-----
1- Binary classification
2- Multiclass classification
3- Multilabel classification

-----Steps in modeling-----
1- Create or import the model
2- Compile the model
3- Fit the model
4- Evaluate the model
5- Tweak 
6- Evaluate...

-----Improving the model-----
1- Add layers
2- Increase the number of hidden units
3- Change the activation function
4- Change the optimization function
5- Change the lr
6- Fitting on more data
7- Fitting for longer
"""

from turtle import circle
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Create a visualization function


def plot_decision_boundary(model, X, Y):
    """Take in a trained model, create a mashgrid with the X values,
    make predictions across the meshgrid and plot the predictions as well 
    as a line betwen zones
    """

    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X value (make predictions on this)
    # Stack 2D arrays together
    x_in = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions
    y_pred = model.predict(x_in)
    # print(len(y_pred[0]))
    # print(y_pred[0])

    # Check for multiclass
    if len(y_pred[0]) > 1:
        print("-----Doing multiclass classification-----")
        # We have to reshape our predictions to make them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("-----Doing binary classification-----")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def Introduction():
    # Create data to view and fit
    n_samples = 1000

    # Make circles
    X, Y = make_circles(n_samples,
                        noise=0.03,
                        random_state=42)

    print(X.shape, Y.shape)

    # Better visualize the data as a df
    circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": Y})
    # print(circles)

    # Plot a graph to understand
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    # plt.show()

    # Set seed
    tf.random.set_seed(42)

    # 1- Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # 2- Compile the model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    # 3- Fit the model
    model.fit(X, Y, epochs=150, verbose=1)

    # 4 & 5- Tweak and evaluate the model
    model.evaluate(X, Y)

    # Check out the predictions our model is making
    plot_decision_boundary(model, X, Y)


Introduction()
