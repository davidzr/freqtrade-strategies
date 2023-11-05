# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class BBRSI2(IStrategy):
    minimal_roi = {
        "0": 0.30,
        "120": 0.20,
        "360": 0.15,
        "720": 0
    }

    stoploss = -0.20

    timeframe = '1m'

    trailing_stop = True

    order_types = {
        "buy": "limit",
        "sell": "limit",
        "emergencysell": "market",
        "forcebuy": "market",
        "forcesell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        # dataframe['bb_upperband'] = bollinger['upper']


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 35)
                & (dataframe['close'] < dataframe['bb_lowerband'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 75)
                & (dataframe['close'] > dataframe['bb_middleband'])
            ),
            'sell'] = 1

        return dataframe
