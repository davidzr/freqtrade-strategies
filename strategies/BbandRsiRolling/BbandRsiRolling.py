# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce

# --------------------------------


class BbandRsiRolling(IStrategy):
    """

    author@: Michael Fourie

    This strategy uses Bollinger Bands and the rolling rsi to determine when it should make a buy.
    Selling is completley determined by the minimal roi.

    """

    # Minimal ROI designed for the strategy.
    # This has been determined through hyperopt in a timerange of 270 days.
    minimal_roi = {
        "0":  0.03279,
        "259": 0.02964,
        "536" : 0.02467,
        "818": 0.02326,
        "965": 0.01951,
        "1230": 0.01492,
        "1279" : 0.01502,
        "1448": 0.00945,
        "1525" : 0.00698,
        "1616": 0.00319,
        "1897" : 0

    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.08

    # Optimal timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']




        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'].rolling(8).min() < 37) &
                    (dataframe['close'] < dataframe['bb_lowerband'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
