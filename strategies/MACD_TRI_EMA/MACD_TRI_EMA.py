# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class MACD_TRI_EMA(IStrategy):
    """

    
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "120": 0.0,
        "30": 0.04,
        "15": 0.06,
        "10": 0.15,
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.03

    # Optimal timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=13)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']) &
                    (dataframe['close'].shift(1) > dataframe['tema'].shift(1)) 

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                   qtpylib.crossed_above(dataframe['macdsignal'], dataframe['macd'])
            ),
            'sell'] = 1
        return dataframe
