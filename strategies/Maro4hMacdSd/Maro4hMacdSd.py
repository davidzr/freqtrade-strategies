# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np # noqa


class Maro4hMacdSd(IStrategy):

    max_open_trades = 1
    stake_amount = 500
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"

    stoploss = -0.21611

    minimal_roi = {
        "0": 0.24627,
        "24": 0.06484,
        "38": 0.02921,
        "145": 0
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.1
    trailing_stop_positive_offset = 0.2

    # run "populate_indicators" only for new candle
    process_only_new_candles = True

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
        # MACD
        macd = ta.MACD(dataframe,fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = 100*macd['macdhist']/dataframe['close']

        dataframe['corr'] = ta.STDDEV(dataframe, timeperiod=28)
        dataframe['corr_mean'] = ta.MA(dataframe['corr'], timeperiod=28)

        dataframe['corr_sell'] = ta.STDDEV(dataframe, timeperiod=28)
        dataframe['corr_mean_sell'] = ta.MA(dataframe['corr'], timeperiod=28)


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
            (dataframe['macdhist'] < 0) &
            (dataframe['macdhist'].shift(2) > dataframe['macdhist'].shift(1))
            & (dataframe['macdhist'] > dataframe['macdhist'].shift(2))
            &
            (dataframe['corr'] > dataframe['corr_mean'])
            ),'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
            (dataframe['macdhist'] > 0) &
            (dataframe['macdhist'].shift(2) < dataframe['macdhist'].shift(1))
            &(dataframe['macdhist'] < dataframe['macdhist'].shift(2)) &
            (dataframe['corr_sell'] < dataframe['corr_mean_sell'])
            ),'sell'] = 1

        return dataframe