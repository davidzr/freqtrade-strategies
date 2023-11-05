
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Roth01(IStrategy):
    # Buy hyperspace params:
    buy_params = {
        'adx-enabled': False,
        'adx-value': 31,
        'cci-enabled': False,
        'cci-value': -74,
        'fastd-enabled': False,
        'fastd-value': 41,
        'mfi-enabled': True,
        'mfi-value': 20,
        'rsi-enabled': False,
        'rsi-value': 34,
        'trigger': 'bb_lower'
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-adx-enabled': True,
        'sell-adx-value': 69,
        'sell-cci-enabled': False,
        'sell-cci-value': 60,
        'sell-fastd-enabled': False,
        'sell-fastd-value': 77,
        'sell-mfi-enabled': True,
        'sell-mfi-value': 92,
        'sell-rsi-enabled': True,
        'sell-rsi-value': 75,
        'sell-trigger': 'sell-bb_upper'
    }

    # ROI table:
    minimal_roi = {
        "0": 0.14696,
        "29": 0.06698,
        "75": 0.02449,
        "181": 0
    }

    # Stoploss:
    stoploss = -0.29585

    # Optimal timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_low'] = bollinger['lower']
        dataframe['bb_mid'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_perc'] = (dataframe['close'] - dataframe['bb_low']) / (
                    dataframe['bb_upper'] - dataframe['bb_low'])
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['mfi'] < 24) &
                (dataframe['close'] < dataframe['bb_low']) &
                (dataframe['cci'] <= -57.0)
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
                (dataframe['sar'] > dataframe['close']) &
                #(dataframe['adx'] > 69) &
                (dataframe['rsi'] > 75) &
                (dataframe['close'] > dataframe['bb_upper']) &
                (dataframe['cci'] >= 83.0) &
                (dataframe['mfi'] < 92) &
                (dataframe['sar'])
                #(dataframe['fastk'] < 51)

            ),
            'sell'] = 1

        return dataframe
