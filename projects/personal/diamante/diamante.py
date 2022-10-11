"""
Predict the price of colorfull diamonds
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


def plotGraph(y_test, y_pred, regressorName):
    plt.scatter(range(len(y_test)), y_test, color='blue', label="Testing data")
    plt.scatter(range(len(y_pred)), y_pred,
                color='red', label="Prediction data")
    plt.title(regressorName)
    plt.show()
    return


def diamond_prices():

    # Importing the database
    data = pd.read_csv('Tabela-diamantes.csv')

    # Defining X and Y
    Y = data['Log Price']
    X = data.drop('Log Price', axis=1)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42)

    # Create the model
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu")])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["mse"])

    # Fit the model
    model.fit(x_train, y_train, epochs=200, verbose=1)

    # Evaluate the model
    model.evaluate(x_test, y_test)

    # Make predictions
    y_pred = model.predict(x_test)

    # Plot the predictions
    plotGraph(y_test, y_pred, "Predictions")


diamond_prices()
