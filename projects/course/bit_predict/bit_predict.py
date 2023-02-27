import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers


class BitPredict():

    def __init__(self, csv_path, date_column_name, horizon, window):
        self.csv_df = pd.read_csv(csv_path, parse_dates=[
                                  date_column_name], index_col=[date_column_name])
        self.date_column_name = date_column_name
        self.horizon = horizon
        self.window = window
        self.prices = 0

    # Create a function to plot time series data
    def _plot_time_series(self, timesteps, values, format='.', start=0, end=None, label=None):
        """
        Plots a timesteps (a series of points in time) against values (a series of values across timesteps).

        Parameters
        ---------
        timesteps : array of timesteps
        values : array of values across time
        format : style of plot, default "."
        start : where to start the plot (setting a value will index from start of timesteps & values)
        end : where to end the plot (setting a value will index from end of timesteps & values)
        label : label to show on plot of values
        """
        # Plot the series
        plt.plot(timesteps[start:end], values[start:end], format, label=label)
        plt.xlabel("Time")
        plt.ylabel("BTC Price")
        if label:
            plt.legend(fontsize=14)  # make label bigger
        plt.grid(True)

    def _clean_data(self, visualize=False):
        closing_prices = pd.DataFrame(self.csv_df["Close"])
        # Split the data into training and testing
        timestamp = closing_prices.index.to_numpy()
        self.prices = closing_prices["Close"].to_numpy()

        # 80% train 20% test
        split_size = int(0.8 * len(self.prices))
        X_train, y_train = timestamp[:split_size], self.prices[:split_size]
        X_test, y_test = timestamp[split_size:], self.prices[split_size:]
        print(f'{len(X_train)} {len(X_test)} {len(y_train)} {len(y_test)}')

        # Plot currently splits
        if visualize:
            plt.figure(figsize=(10, 7))
            self._plot_time_series(
                timesteps=X_train, values=y_train, label="Train data")
            self._plot_time_series(
                timesteps=X_test, values=y_test, label="Test data")
            plt.show()

        # Return splitted date
        return X_train, y_train, X_test, y_test

    def _mean_absolute_scaled_error(self, y_true, y_pred):

        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

        return mae / mae_naive_no_season

    def _evaluate_preds(self, y_true, y_pred):

        # Change dt for metric calculations
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Calculate various metrics
        mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
        mse = tf.keras. metrics.mean_squared_error(y_true, y_pred)
        rmse = tf.sqrt(mse)
        mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
        mase = self._mean_absolute_scaled_error(y_true, y_pred)

        return {
            "mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()
        }

    def _naive_forecast_baseline(self, visualize=False):

        # Split data
        X_train, y_train, X_test, y_test = self._clean_data()

        # Create naive forecast
        naive_forecast = y_test[:-1]

        # Plot the forecast
        if visualize:
            plt.figure(figsize=(10, 7))
            self._plot_time_series(
                timesteps=X_train, values=y_train, label="Train data")
            self._plot_time_series(
                timesteps=X_test, values=y_test, label="Test data")
            self._plot_time_series(
                timesteps=X_test[1:], values=naive_forecast, format="-", label="Naive forecast")
            plt.show()

        # Evaluate preds
        naive_results = self._evaluate_preds(
            y_true=y_test[1:], y_pred=naive_forecast)
        print(naive_results)

    def _get_labeled_windows(self, x):

        return x[:, :-self.horizon], x[:, -self.horizon:]

    def _make_windows(self, x):

        window_step = np.expand_dims(
            np.arange(self.window + self.horizon), axis=0)
        window_indexes = window_step + \
            np.expand_dims(
                np.arange(len(x)-(self.window+self.horizon-1)), axis=0).T
        windowed_array = x[window_indexes]
        windows, labels = self._get_labeled_windows(windowed_array)

        return windows, labels

    def _make_train_test_splits(self, windows, labels, test_split=0.2):
        """
        Splits matching pairs of windows and labels into train and test splits.
        """
        split_size = int(len(windows) * (1-test_split))
        train_window = windows[:split_size]
        train_labels = labels[:split_size]
        test_windows = windows[split_size:]
        test_labels = labels[split_size:]
        return train_window, test_windows, train_labels, test_labels

    def _create_model_checkpoint(self, model_name, save_path="model_experiments"):
        """
        Saves the best performance epoch of each tested model.
        """
        return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                                  verbose=0,
                                                  save_best_only=True)

    def main(self):
        self._naive_forecast_baseline(visualize=False)
        full_windows, full_labels = self._make_windows(self.prices)
        # Split matching pairs of windows and labels
        train_windows, test_windows, train_labels, test_labels = self._make_train_test_splits(
            full_windows, full_labels)
        print(
            f'{len(train_windows)} {len(test_windows)} {len(train_labels)} {len(test_labels)}')


if __name__ == "__main__":
    df = BitPredict('PETR4_1d.csv', 'Date', 1, 7)
    df.main()
