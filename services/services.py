from abc import abstractclassmethod
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd


class MachineLearningMetrics:

    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_decision_boundary(model, x_test, y_test):
        """Take in a trained model, create a mashgrid with the X values,
        make predictions across the meshgrid and plot the predictions as well
        as a line betwen zones
        """

        # Define the axis boundaries of the plot and create a meshgrid
        x_min, x_max = x_test[:, 0].min() - 0.1, x_test[:, 0].max() + 0.1
        y_min, y_max = x_test[:, 1].min() - 0.1, x_test[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # Create X value (make predictions on this)
        # Stack 2D arrays together
        x_in = np.c_[xx.ravel(), yy.ravel()]

        # Make predictions
        y_pred = model.predict(x_in)

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
        plt.scatter(x_test[:, 0], x_test[:, 1],
                    c=y_test, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()

    @staticmethod
    def pretty_confusion_matrix(y_test, y_pred):
        """
        Plot the confusion matrix to better visualize the model
        """
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

    @staticmethod
    def plot_loss_curve(history):
        """
        Plot the loss curve to see the realtionship between the history variables
        """
        pd.DataFrame(history.history).plot(xlabel="epochs")
        plt.title("Model training curve")
        plt.show()

    @staticmethod
    def plot_loss_vs_learning(history):
        """
        Find the ideal learning rate: where the loss istill decreasing, but not flatten out
        """
        lrs = 1e-4 * (10 ** (np.arange(100)/20))
        plt.figure(figsize=(10, 7))
        # we want the x-axis (learning rate) to be log scale
        plt.semilogx(lrs, history.history["loss"])
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning rate vs. loss")
        plt.show()
