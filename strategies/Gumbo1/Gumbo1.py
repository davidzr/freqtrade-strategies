"""
https://github.com/raph92?tab=repositories
"""
import logging

# --- Do not remove these libs ---
import sys
from functools import reduce
from pathlib import Path

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.strategy import (
    IntParameter,
    DecimalParameter,
    merge_informative_pair,
)
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)


class Gumbo1(IStrategy):
    # region Parameters
    ewo_low = DecimalParameter(-20.0, 1, default=0, space="buy", optimize=True)
    t3_periods = IntParameter(5, 20, default=5, space="buy", optimize=True)

    stoch_high = IntParameter(60, 100, default=80, space="sell", optimize=True)
    stock_periods = IntParameter(70, 90, default=80, space="sell", optimize=True)

    # endregion
    # region Params
    minimal_roi = {"0": 0.10, "20": 0.05, "64": 0.03, "168": 0}
    stoploss = -0.25
    # endregion
    timeframe = '5m'
    use_custom_stoploss = False
    inf_timeframe = '1h'
    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True
    startup_candle_count = 200

    def informative_pairs(self) -> ListPairsWithTimeframes:
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_informative_indicators(self, dataframe: DataFrame, metadata):
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.inf_timeframe
        )
        # t3 from custom_indicators
        informative['T3'] = T3(informative)
        # bollinger bands
        bbands = ta.BBANDS(informative, timeperiod=20)
        informative['bb_lowerband'] = bbands['lowerband']
        informative['bb_middleband'] = bbands['middleband']
        informative['bb_upperband'] = bbands['upperband']

        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, self.inf_timeframe
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ewo
        dataframe['EWO'] = EWO(dataframe)
        # ema
        dataframe['EMA'] = ta.EMA(dataframe)
        # t3
        for i in self.t3_periods.range:
            dataframe[f'T3_{i}'] = T3(dataframe, i)
        # bollinger bands 40
        bbands = ta.BBANDS(dataframe, timeperiod=40)
        dataframe['bb_lowerband_40'] = bbands['lowerband']
        dataframe['bb_middleband_40'] = bbands['middleband']
        dataframe['bb_upperband_40'] = bbands['upperband']
        # stochastic
        # stochastic windows
        for i in self.stock_periods.range:
            dataframe[f'stoch_{i}'] = stoch_sma(dataframe, window=i)
        dataframe = self.populate_informative_indicators(dataframe, metadata)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # ewo < 0
        conditions.append(dataframe['EWO'] < self.ewo_low.value)
        # middleband 1h >= t3 1h
        conditions.append(dataframe['bb_middleband_1h'] >= dataframe['T3_1h'])
        # t3 <= ema
        conditions.append(dataframe[f'T3_{self.t3_periods.value}'] <= dataframe['EMA'])
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        # stoch > 80
        conditions.append(
            dataframe[f'stoch_{self.stock_periods.value}'] > self.stoch_high.value
        )
        # t3 >= middleband_40
        conditions.append(
            dataframe[f'T3_{self.t3_periods.value}'] >= dataframe['bb_middleband_40']
        )
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'sell'] = 1
        return dataframe


def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["low"] * 100
    return emadif


def stoch_sma(dataframe: DataFrame, window=80):
    """"""
    stoch = qtpylib.stoch(dataframe, window)
    return qtpylib.sma((stoch['slow_k'] + stoch['slow_d']) / 2, 10)
