import talib.abstract as ta
import numpy as np  # noqa
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter

__author__ = "Robert Roman"
__copyright__ = "Free For Use"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Robert Roman"
__email__ = "robertroman7@gmail.com"
__BTC_donation__ = "3FgFaG15yntZYSUzfEpxr5mDt1RArvcQrK"


# Optimized With Sharpe Ratio and 1 years data
# 12520 trades. 6438/5337/745 Wins/Draws/Losses. Avg profit   1.55%. Median profit   0.17%. Total profit  194026.95473822 USDT ( 194.03%). Avg duration 1 day, 9:13:00 min. Objective: -63.61104

class Trend_Strength_Directional(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '15m'

    # ROI table:
    minimal_roi = {
        "0": 0.383,
        "120": 0.082,
        "283": 0.045,
        "495": 0
    }

    # Stoploss:
    stoploss = -0.314

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.307
    trailing_stop_positive_offset = 0.364
    trailing_only_offset_is_reached = False

    # Hyperopt Buy Parameters
    buy_plusdi_enabled = CategoricalParameter([True, False], space='buy', optimize=True, default=False)
    buy_adx = IntParameter(low=1, high=100, default=12, space='buy', optimize=True, load=True)
    buy_adx_timeframe = IntParameter(low=1, high=50, default=9, space='buy', optimize=True, load=True)
    buy_plusdi = IntParameter(low=1, high=100, default=44, space='buy', optimize=True, load=True)
    buy_minusdi = IntParameter(low=1, high=100, default=74, space='buy', optimize=True, load=True)

    # Hyperopt Sell Parameters
    sell_plusdi_enabled = CategoricalParameter([True, False], space='sell', optimize=True, default=True)
    sell_adx = IntParameter(low=1, high=100, default=3, space='sell', optimize=True, load=True)
    sell_adx_timeframe = IntParameter(low=1, high=50, default=41, space='sell', optimize=True, load=True)
    sell_plusdi = IntParameter(low=1, high=100, default=49, space='sell', optimize=True, load=True)
    sell_minusdi = IntParameter(low=1, high=100, default=11, space='sell', optimize=True, load=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # GUARDS
        if self.buy_plusdi_enabled.value:
            conditions.append(ta.PLUS_DI(dataframe, timeperiod=int(self.buy_plusdi.value)) > ta.MINUS_DI(dataframe, timeperiod=int(self.buy_minusdi.value)))

        # TRIGGERS
        try:
            conditions.append(ta.ADX(dataframe, timeperiod=int(self.buy_adx_timeframe.value)) > self.buy_adx.value)
        except Exception:
            pass

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # GUARDS
        if self.sell_plusdi_enabled.value:
            conditions.append(ta.PLUS_DI(dataframe, timeperiod=int(self.sell_plusdi.value)) < ta.MINUS_DI(dataframe, timeperiod=int(self.sell_minusdi.value)))

        # TRIGGERS
        try:
            conditions.append(ta.ADX(dataframe, timeperiod=int(self.sell_adx_timeframe.value)) < self.sell_adx.value)
        except Exception:
            pass

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
