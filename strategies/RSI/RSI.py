# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair, informative
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
import talib.abstract as ta
from talib import WILLR
import freqtrade.vendor.qtpylib.indicators as qtpylib
import requests
import json
# --------------------------------




class RSI(IStrategy):


    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0":  0.09
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.99
    # custom_stop = -0.1
    # Optimal timeframe for the strategy
    timeframe = '15m'


    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def informative_pairs(self):
        return []



    @informative('30m')
    def populate_indicators_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rperc'] = ta.WILLR(dataframe, timeperiod=14)
        return dataframe



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rperc'] = ta.WILLR(dataframe, timeperiod=14)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        # rsi_cond = dataframe['rsi_15m'].iloc[-1] <30 and dataframe['rsi_15m'].iloc[-2]<30
        dataframe.loc[(dataframe['rsi']<30) & (dataframe['rperc']<-80),'buy'] = 1
        # dataframe.loc[(dataframe['rsi']<=30) & (dataframe['rsi'].iloc[-1] <=30)] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[(dataframe['rsi_30m']>70) &
                            (dataframe['rperc_30m']>-20) ,'sell'] = 1


        return dataframe
