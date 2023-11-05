from pandas import DataFrame
import talib.abstract as ta
from functools import reduce
import numpy as np
from freqtrade.strategy import (IStrategy)

# DIV v1.0 - 2021-09-07
# by Sanka 

class DIV_v1(IStrategy):

    minimal_roi = {
        "0": 0.10347601757573865,
        "3": 0.050495605759981035,
        "5": 0.03350898081823659,
        "61": 0.0275218557571848,
        "292": 0.005185372158403069,
        "399": 0,
        
    }
    stoploss = -0.15

    timeframe = '5m'
    startup_candle_count = 200
    process_only_new_candles = True

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    plot_config = {
        "main_plot": {
            "ohlc_bottom" : {
                "type": "scatter",
                'plotly': {
                    "mode": "markers",
                    "name": "a",
                    "text": "aa",
                    "marker": {
                        "symbol": "cross-dot",
                        "size": 3,
                        "color": "black"
                    }
                }
            },
        },
        "subplots": {
            "rsi": {
                "rsi": {"color": "blue"},
                "rsi_bottom" : {
                    "type": "scatter",
                    'plotly': {
                        "mode": "markers",
                        "name": "b",
                        "text": "bb",
                        "marker": {
                            "symbol": "cross-dot",
                            "size": 3,
                            "color": "black"
                        }
                    }
                },
            }
        }
    }

    #############################################################

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Divergence
        dataframe = divergence(dataframe, "rsi")

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["bullish_divergence"] == True) &
                (dataframe['rsi'] < 30) &
                (dataframe["volume"] > 0)
            ), 'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe


def divergence(dataframe: DataFrame, source='rsi'):
    # Detect divergence between close price and source

    # Detect HL or LL
    dataframe['ohlc_bottom'] = np.NaN
    dataframe['rsi_bottom'] = np.NaN
    dataframe.loc[(dataframe['close'].shift() <= dataframe['close'].shift(2)) & (dataframe['close'] >= dataframe['close'].shift()), 'ohlc_bottom'] = dataframe['close'].shift()
    dataframe.loc[(dataframe[source].shift() <= dataframe[source].shift(2)) & (dataframe[source] >= dataframe[source].shift()), 'rsi_bottom'] = dataframe[source].shift()
    dataframe["ohlc_bottom"].fillna(method='ffill', inplace=True)
    dataframe["rsi_bottom"].fillna(method='ffill', inplace=True)

    # Detect divergence
    dataframe['bullish_divergence'] = np.NaN
    dataframe['hidden_bullish_divergence'] = np.NaN
    for i in range(2, 15):
        # Check there is nothing between the 2 diverging points
        conditional_array = []
        for ii in range(1, i):
            conditional_array.append(dataframe["ohlc_bottom"].shift(i).le(dataframe['ohlc_bottom'].shift(ii)))
        res = reduce(lambda x, y: x & y, conditional_array)
        dataframe.loc[(
            (dataframe["ohlc_bottom"].lt(dataframe['ohlc_bottom'].shift(i))) &
            (dataframe["rsi_bottom"].gt(dataframe['rsi_bottom'].shift(i))) &
            (dataframe["ohlc_bottom"].le(dataframe['ohlc_bottom'].shift())) &
            (res)
            ), "bullish_divergence"] = True

    return dataframe
