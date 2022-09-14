"""
-----Predict the RSI using the btc price-----
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
