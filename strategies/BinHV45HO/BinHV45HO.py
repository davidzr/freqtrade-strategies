# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter
import numpy as np
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)

    return rolling_mean, lower_band


class BinHV45HO(IStrategy):
    buy_params = {
        "df_close_bbdelta": 0.057,
        "df_close_closedelta": 0.016,
        "df_tail_bbdelta": 0.293,
    }

    minimal_roi = {
        "0": 0.0125
    }

    stoploss = -0.19
    timeframe = '1m'

    df_close_bbdelta = DecimalParameter(0.005, 0.06, default=0.008, space='buy', optimize=False, load=True)
    df_close_closedelta = DecimalParameter(0.01, 0.03, default=0.0175, space='buy', optimize=False, load=True)
    df_tail_bbdelta = DecimalParameter(0.15, 0.45, default=0.25, space='buy', optimize=False, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['mid'] = np.nan_to_num(mid)
        dataframe['lower'] = np.nan_to_num(lower)
        dataframe['bbdelta'] = (dataframe['mid'] - dataframe['lower']).abs()
        dataframe['pricedelta'] = (dataframe['open'] - dataframe['close']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * self.df_close_bbdelta.value) &
                dataframe['closedelta'].gt(dataframe['close'] * self.df_close_closedelta.value) &
                dataframe['tail'].lt(dataframe['bbdelta'] * self.df_tail_bbdelta.value) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift())
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        no sell signal
        """
        dataframe.loc[:, 'sell'] = 0
        return dataframe
