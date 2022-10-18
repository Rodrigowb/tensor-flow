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

-----Non Linear activation functions (deduce paterns in non-linear data)-----
1- Sigmoid: 1 / 1 + exp(-x)
2- Relu: max(x, 0)

-----Linear activation functions-----
1- Linear: does not change the input data

-----Classification evaluation methods-----
1- Accuracy: default for classification
2- Precision
3- Recall
Obs: tradeoff between precision and recall
4- F1score: combination of precision and recall
5- Confusion matrix

"""

from gc import callbacks
from turtle import circle
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
from services import MachineLearningMetrics

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


def pretty_confusion_matrix(y_test, y_pred):
    figsize = (10, 10)

    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / \
        cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    # https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.matshow.html
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Create classes
    classes = False

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(20)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=15)
    #  Plot the graph
    plt.show()


def plot_loss_curve(history):
    pd.DataFrame(history.history).plot(xlabel="epochs")
    plt.title("Model training curve")
    plt.show()


def plot_loss_vs_learning(history):
    """Find the ideal learning rate: where the loss istill decreasing, but not flatten out"""
    lrs = 1e-4 * (10 ** (np.arange(100)/20))
    plt.figure(figsize=(10, 7))
    # we want the x-axis (learning rate) to be log scale
    plt.semilogx(lrs, history.history["loss"])
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs. loss")
    plt.show()


def Introduction():
    # Create data to view and fit
    n_samples = 1000

    # Make circles
    X, Y = make_circles(n_samples,
                        noise=0.03,
                        random_state=42)

    # Better visualize the data as a df
    circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": Y})

    # Plot a graph to understand
    plt.scatter(X[:, 0], X[:, 1], c=Y)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    # # Set seed
    tf.random.set_seed(42)

    # 1- Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # 2- Compile the model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  metrics=["accuracy"])

    # Create a learning rate scheduler callback
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-4 * 10**(epoch/20))

    # 3- Fit the model
    history = model.fit(x_train, y_train, epochs=100,
                        verbose=1, callbacks=[lr_scheduler])

    # 4 & 5- Tweak and evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Model loss on the test set: {loss}")
    print(f"Model accuracy on the test set: {100*accuracy:.2f}%")

    # Make predictions
    prediction = model.predict(x_test)

    # Plot the loss curve (how the model is behaving while learning?)
    plot_loss_curve(history)
    plot_loss_vs_learning(history)
    print(pd.DataFrame(history.history))

    # Transform the predictions into binary
    prediction_binary = tf.round(prediction)

    # Create a confusion matrix
    pretty_confusion_matrix(y_test, prediction_binary)

    # Check out the predictions our model is making
    # plot_decision_boundary(model, x_test, y_test)
    # mlm.plot_decision_boundary(model, x_test, y_test)


Introduction()
