# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class ONUR(IStrategy):
    minimal_roi = {
        "0": 0.131,
        "109": 0.08,
        "226": 0.03
    }

    stoploss = -0.99
    trailing_stop = True
    trailing_stop_positive = 0.293
    trailing_stop_positive_offset = 0.362
    trailing_only_offset_is_reached = True

    timeframe = '15m'

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
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 74)
                & (dataframe['close'] > dataframe['bb_middleband'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # (dataframe['close'] > dataframe['bb_upperband'])
            ),
            'sell'] = 1

        return dataframe