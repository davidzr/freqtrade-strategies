# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series, DatetimeIndex, merge
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Cci(IStrategy):
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.18967,
        "28": 0.07749,
        "72": 0.03823,
        "180": 0
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.23

    # Optimal timeframe for the strategy
    timeframe = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.resample(dataframe, self.timeframe, 5)

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=18)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (dataframe['cci'] < -198)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (dataframe['cci'] > 197)
            ),
            'sell'] = 1
        return dataframe


    def resample(self, dataframe, interval, factor):
        # defines the reinforcement logic
        # resampled dataframe to establish if we are in an uptrend, downtrend or sideways trend
        df = dataframe.copy()
        df = df.set_index(DatetimeIndex(df['date']))
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        df = df.resample(str(int(interval[:-1]) * factor) + 'min', label="right").agg(ohlc_dict)
        df['resample_sma'] = ta.SMA(df, timeperiod=100, price='close')
        df['resample_medium'] = ta.SMA(df, timeperiod=50, price='close')
        df['resample_short'] = ta.SMA(df, timeperiod=25, price='close')
        df['resample_long'] = ta.SMA(df, timeperiod=200, price='close')
        df = df.drop(columns=['open', 'high', 'low', 'close'])
        df = df.resample(interval[:-1] + 'min')
        df = df.interpolate(method='time')
        df['date'] = df.index
        df.index = range(len(df))
        dataframe = merge(dataframe, df, on='date', how='left')
        return dataframe