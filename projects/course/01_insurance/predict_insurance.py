"""
-----How long should we train a model?-----
Depends, but TF has a solution, called EarlyStoppingCallback:
It is a TF component we can add to the model to stop training once it stops improving
a certain metric.
"""

from pickletools import optimize
from tabnanny import verbose
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read the dataset
insurance = pd.read_csv('./insurance.csv')

# One hot encode non numerical variables
insurance_one_hot = pd.get_dummies(insurance)

# Create X and Y labels (features and labels)
X = insurance_one_hot.drop("charges", axis=1)
Y = insurance_one_hot["charges"]

# Create a training and test set
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Build a neural network (x_train and y_train)
tf.random.set_seed(42)

# 1- Create a model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

# 2- Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["mae"]
                        )

# 3- Fit the model
history = insurance_model.fit(tf.expand_dims(x_train, axis=-1),
                              y_train, epochs=100, verbose=1)

# 4- Check the results of the insurance model on the test data
insurance_model.evaluate(x_test, y_test)

# 5- Try to improve the model

# Plot history (also know as loss curve or training curve)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
# plt.show()
