# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta


class MACDRSI200(IStrategy):

    ticker_interval = '5m'

    # ROI table:
    minimal_roi = {
        "0": 0.03024,
        "296": 0.02924,
        "596": 0.02545,
        "840": 0.02444,
        "966": 0.02096,
        "1258": 0.01709,
        "1411": 0.01598,
        "1702": 0.0122,
        "1893": 0.00732,
        "2053": 0.00493,
        "2113": 0
    }

    # Stoploss:
    stoploss = -0.04032

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        macd = ta.MACD(dataframe, fastperiod=24, slowperiod=56, signalperiod=6)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['sell-rsi'] = ta.RSI(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'].rolling(8).min() < 41) &
                    (dataframe['close'] > dataframe['ema200']) &
                    (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'].rolling(8).max() > 93) &
                    (dataframe['macd'] > 0) &
                    (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))
            ),
            'sell'] = 1

        return dataframe
