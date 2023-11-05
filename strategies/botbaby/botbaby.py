# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class botbaby(IStrategy):

    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2
    # IMP NOTEEEEEEEE - also change this roi parameter after testing
    minimal_roi = {

        "0": 0.01
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    #  IMP NOTE -Hey listen, remeber to change stoploss after testing
    stoploss = -0.007

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = '30m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }


    def informative_pairs(self):

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # ADX


         # EMA - Exponential Moving Average

         dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
         dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
         dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
         dataframe['ema13'] = ta.EMA(dataframe, timeperiod = 13 )
         dataframe['ema50'] = ta.EMA(dataframe, timeperiod = 50)

         return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (
                (dataframe['ema13'] > dataframe['ema50']) &
                (dataframe['ema13'].shift(1) <= dataframe['ema50'].shift(1))
                )
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (
                (dataframe['ema13'] < dataframe['ema50'])
                )
            ),
            'sell'] = 1

        return dataframe