# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
import numpy as np  # noqa
import pandas as pd  # noqa

# These libs are for hyperopt
from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter)

class keltnerchannel(IStrategy):
    timeframe = "6h"
    # Both stoploss and roi are set to 100 to prevent them to give a sell signal.
    stoploss = -0.254
    minimal_roi = {"0": 100}

    plot_config = {
        "main_plot": {
            "kc_upperband" : {"color": "purple",'plotly': {'opacity': 0.4}},
            "kc_middleband" : {"color": "blue"},
            "kc_lowerband" : {"color": "purple",'plotly': {'opacity': 0.4}}
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "orange"},
                "hline": {"color": "grey","plotly": {"opacity": 0.4}}
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe, window=20, atrs=1)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Horizontal RSI line
        hline = 55
        dataframe['hline'] = hline

        # Print stuff for debugging dataframe
        # print(metadata)
        # print(dataframe.tail(20))
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_above(dataframe['close'], dataframe['kc_upperband'])
            & (dataframe["rsi"] > dataframe['hline'])
            ),

            "buy",
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_below(dataframe['close'], dataframe['kc_middleband'])),

            "sell",
        ] = 1
        return dataframe
