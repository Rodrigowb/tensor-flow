"""
-----NeuralNetwork Regression Exercises-----
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


def ExerciseOne():
    """Create our own dataset and make a neural network regression model with it"""

    # Create X and Y
    def CreateX():
        X = list()
        for number in range(0, 501):
            X.append(number)
        return X

    def CreateY():
        X = CreateX()
        Y = list()
        for number in X:
            Y.append(number*10 + 10)
        return Y

    X = pd.DataFrame(CreateX())
    Y = pd.DataFrame(CreateY())

    # Split the data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    # Create the model
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        # tf.keras.layers.Dense(100, activation="relu"),
        # tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.Adam(lr=0.001, epsilon=0.1),
                  metrics=["mae"])

    # Fit the model
    history = model.fit(tf.expand_dims(x_train, axis=-1),
                        y_train, epochs=200, verbose=1)

    # Evaluate the model
    model.evaluate(x_test, y_test)

    # Plot the history
    pd.DataFrame(history.history).plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.show()


ExerciseOne()
