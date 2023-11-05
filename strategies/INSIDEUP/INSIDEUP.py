import copy
import logging
import pathlib
import rapidjson
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, timeframe_to_minutes
from freqtrade.exchange import timeframe_to_prev_date
from pandas import DataFrame, Series, concat
from functools import reduce
import math
from typing import Dict
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from technical.util import resample_to_interval, resampled_merge
from technical.indicators import RMI, zema, VIDYA, ichimoku
import time
import warnings

# Custom imports to fetch API data
import requests
import json


class INSIDEUP(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.237,
        "4195": 0.17,
        "7191": 0.053,
        "14695": 0
    }

    # Stoploss:
    stoploss = -0.99  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.011  # value loaded from strategy
    trailing_stop_positive_offset = 0.029  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    # Optimal timeframe for the strategy.
    timeframe = '1d'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Inputs:
        # # prices: ['open', 'high', 'low', 'close']

        # # Three Inside Up/Down: values [0, -100, 100]
        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [-100, 0, 100]
        # # MORNINGDOJISTAR: values [0, 100]
        dataframe['CDLMORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(dataframe) # values [0, 100]
        # # Piercing Line: values [0, 100]
        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Three Black Crows: values [-100, 0, 100]
        dataframe['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(dataframe) # values [-100, 0, 100]

        # RSI
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['slowadx'] = ta.ADX(dataframe, 35)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dateTime = datetime.now()

        dataframe.loc[
            # Check for downtrend movement
            (
             (dataframe['close'] < dataframe['close'].shift(2)) |
             (dataframe['rsi_14'] < 50)
            ) &
            (
             (dataframe['adx'] > 13.0)
            ) &
            # Check for patterns
            (
                # the user should consider that a three inside up is significant
                #when it appears in a downtrend
                (dataframe['CDL3INSIDE'] >= 0).any() # Bullish
            ),
            ['buy', 'buy_tag']] = (1, 'buy_3_inside')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        no sell signal
        """
        dataframe.loc[:, 'sell'] = 0
        return dataframe
