"""
-----Objective-----
Develop a regression model that predicts the price of the next day BTC according to the SPY price.

-----Steps-----
1- Download the daily data (2017-2022)- OK
2- Make a DF with the close price of SPY (X- independente) and BTC (Y- dependent)- Ok
3- Remove the Nan rows- Ok
4- Project the BTC price 1 day further (delete the Nan generated row)- Ok
5- Transform the data into TENSORS- Ok
6- Split the data (X_train, Y_train, X_test, Y_test)- Ok
7- Create the model- Ok
8- Compile the model Ok
9- Fith the model
10- Make predictions
"""
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# ----------Manipulating the data-----------

# Import the data and get the corret data range
X_spy = pd.read_csv('SPY.csv', usecols=[0, 4])
date_range = X_spy['Date']

# Select the BTC data according to the data_range
Y_btc = pd.read_csv('BTC-USD.csv', usecols=[0, 4])
print(Y_btc)

# Gent only the matching data
Y_btc_filtered = Y_btc.loc[Y_btc['Date'].isin(date_range.values)]

# Rename the columns
X_spy.rename(columns={'Date': 'Date', 'Close': 'Spy Close'}, inplace=True)
Y_btc_filtered.rename(
    columns={'Date': 'Date', 'Close': 'Btc Close'}, inplace=True)

# Add a column to the X_spy df and making it as the final one
X_spy['Btc Close'] = Y_btc_filtered['Btc Close'].values
df = X_spy
print(df)

# Checking Nan data in the df
print(df.isnull().sum())

# ----------Shift the BTC data 1 day forward-----------

# Shift the BTC data 1 period further
df['Btc Close'] = df['Btc Close'].shift(periods=1)

# Remove the Nan
df.dropna(inplace=True)
print(df)

# ----------Transform the data into tensors-----------
X = tf.constant(df['Spy Close'])
Y = tf.constant(df['Btc Close'])
print(X)
print(Y)

# ----------Split the data into train and test-----------
# Splitting the data
train_test_limit = math.floor(len(X) * 0.8)
X_train = X[:train_test_limit]
Y_train = Y[:train_test_limit]
X_test = X[train_test_limit:]
Y_test = Y[train_test_limit:]

# Visualizing the data
plt.figure(figsize=(10, 7))
plt.scatter(X_train, Y_train, c="b", label="Training data")
plt.scatter(X_test, Y_test, c="g", label="Testing data")
plt.legend()
# plt.show()

# ----------Creating the model-----------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        100, input_shape=[1], name="input_layer", activation="relu"),
    tf.keras.layers.Dense(
        100, input_shape=[1], name="hl_one", activation="relu"),
    tf.keras.layers.Dense(
        100, input_shape=[1], name="hl_two", activation="relu"),
    tf.keras.layers.Dense(1, name="output_layer")
])

# ----------Compiling the model-----------
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=["mae"])

# ----------Fit the model-----------
model.fit(tf.expand_dims(X_train, axis=-1),
          Y_train, epochs=100, verbose=1)

# ----------Predict and plot the predictions-----------


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


# Make predictions using the model
Y_pred = model.predict(X_test)
plot_predictions(train_data=X_train,
                 train_labels=Y_train,
                 test_data=X_test,
                 test_labels=Y_test,
                 predictions=Y_pred)
plt.show()
