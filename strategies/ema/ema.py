# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np# noqa


class ema(IStrategy):

    max_open_trades = 10
    stake_amount = 50
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -1

    minimal_roi = {
        "0": 10
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.1
    trailing_stop_positive_offset = 0.2

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        dataframe['ema6'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema24'] = ta.EMA(dataframe, timeperiod=18)

        dataframe['ema11'] = ta.EMA(dataframe, timeperiod=32)
        dataframe['ema25'] = ta.EMA(dataframe, timeperiod=64)

        dataframe['ema'] =dataframe['ema6']-dataframe['ema24']
        dataframe['ema2'] = dataframe['ema11'] - dataframe['ema25']

        dataframe['ema']= dataframe['ema']*0.6 + dataframe['ema2']*0.5
        dataframe['ema2'] = ta.SMA(dataframe['ema'], timeperiod=29)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
            (qtpylib.crossed_above(dataframe['ema'],dataframe['ema2']))
            ),'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[(qtpylib.crossed_below(dataframe['ema'], dataframe['ema2'])),'sell'] = 1

        return dataframe