from asyncio import constants
from cgi import test
import os
from decouple import config
from binance import Client
from constants import klines
import pandas as pd
import datetime as dt
import ccxt


class BinanceIntegration():
    """Connects to the binance API to get data and fetch orders"""

    API_KEY = config('API_KEY')
    API_SECRET = config('API_SECRET')

    def __init__(self, sandbox=True):
        self.sandbox = sandbox
        self.client = None

    def authenticate_client(self):
        """Authenticate with binance API"""
        if self.sandbox == True:
            try:
                self.client = Client(
                    self.API_KEY, self.API_SECRET, testnet=self.sandbox)
                print(self.client.get_system_status())
                return print("Connected to the SANDBOX")
            except:
                return print("ERROR to connect to the SANDBOX")
        else:
            try:
                self.client = Client(
                    self.API_KEY, self.API_SECRET, testnet=self.sandbox)
                print(self.client.get_system_status())
                return print("Connected to PRODUCTION")
            except:
                return print("ERROR to connect to PRODUCTION")

    def get_historical_data(self, symbol, timeframe, limit):
        binance = ccxt.binance({
            'enableRateLimit': False,
            'apiKey': self.API_KEY,
            'secret': self.API_SECRET
        })
        bars = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            bars, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        return df


if __name__ == "__main__":
    client = BinanceIntegration(sandbox=True)
    data = client.get_historical_data(
        "BTCUSDT", "15m", 5000)
    print(data)
