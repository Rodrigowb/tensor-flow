"""
-----Predict the RSI using the btc price-----
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas_datareader as pdr
import datetime as dt


def PredictRSI():
    # Import the data
    data = pd.read_csv('./BTC-USD-MAX.csv', usecols=[0, 4])

    # Calculate the price difference
    data['Delta'] = data['Close'].diff(1)
    data.dropna(inplace=True)

    # Get the positive and negative numbers
    positive = data['Delta'].copy()
    negative = data['Delta'].copy()
    positive[positive < 0] = 0
    negative[negative > 0] = 0
    data['Positive'] = positive
    data['Negative'] = negative

    # Calculate the RS
    days = 14
    average_gain = positive.rolling(window=days).mean()
    average_loss = abs(negative.rolling(window=days).mean())
    relative_strength = average_gain / average_loss
    data['RS'] = relative_strength

    # Calculate the RSI
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))
    data['RSI'] = RSI
    data.dropna(inplace=True)

    # Defining X and Y
    X = data['Close']
    Y = data['RSI']

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    # Create the model
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation="relu"),
        # tf.keras.layers.Dense(6, activation="relu"),
        # tf.keras.layers.Dense(3, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["mae"])

    # Fit the model
    history = model.fit(tf.expand_dims(x_train, axis=-1),
                        y_train, epochs=500, verbose=1)
    # Evaluate the model
    model.evaluate(x_test, y_test)


def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.show()


def plotGraph(y_test, y_pred, regressorName):
    plt.scatter(range(len(y_test)), y_test, color='blue', label="Testing data")
    plt.scatter(range(len(y_pred)), y_pred,
                color='yellow', label="Prediction data")
    plt.title(regressorName)
    plt.show()
    return


def PredictBinary():
    # Import the data
    data = pd.read_csv('./BTC-USD-MAX.csv', usecols=[0, 4])

    # Calculate the price difference
    data['Delta'] = data['Close'].diff(1)
    data.dropna(inplace=True)

    # Get the positive and negative numbers
    positive = data['Delta'].copy()
    negative = data['Delta'].copy()
    positive[positive < 0] = 0
    negative[negative > 0] = 0
    data['Positive'] = positive
    data['Negative'] = negative

    # Calculate the RS
    days = 14
    average_gain = positive.rolling(window=days).mean()
    average_loss = abs(negative.rolling(window=days).mean())
    relative_strength = average_gain / average_loss
    data['RS'] = relative_strength

    # Calculate the RSI
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))
    data['RSI'] = RSI

    # Calculate the MACD
    data['exp1'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['exp2'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['exp1'] - data['exp2']
    data['exp3'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger bands
    data['sma'] = data['Close'].rolling(15).mean()
    data['std'] = data['Close'].rolling(15).std()
    data['BUP'] = data['sma'] + data['std'] * 2
    data['BDOWN'] = data['sma'] - data['std'] * 2

    # Calculating ema 8
    data['exp8'] = data['Close'].ewm(span=21, adjust=False).mean()

    # Drop Nan numbers
    data.dropna(inplace=True)

    # Insert 0 or 1 according to the changes
    data['Result'] = 1
    data.loc[data['Delta'] < 0, 'Result'] = 0

    # Defining X and Y
    X = data[['RSI', 'MACD', 'exp8']]
    Y = data['Result']

    # Check if output has only 0 and 1
    # print(Y.unique())
    # print(data)

    # Print X
    # print(X)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    # Create the model
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(75, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                  metrics=["accuracy"])

    # Fit the model
    model.fit(x_train, y_train, epochs=100, verbose=1)

    # Evaluate the model
    model.evaluate(x_test, y_test)

    # Make predictions
    y_pred = model.predict(x_test)

    # Covert prediction array into 0 and 1 from predictions probability
    y_pred_binary = tf.round(y_pred)

    # Plot the predictions
    plotGraph(y_test, y_pred_binary, "Predictions")


PredictBinary()
