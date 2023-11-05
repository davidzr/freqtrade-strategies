# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series, DatetimeIndex, merge
from datetime import datetime, timedelta
# --------------------------------

import talib.abstract as ta
import numpy as np
import pandas_ta as pta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
import freqtrade.vendor.qtpylib.indicators as qtpylib

class KC_BB(IStrategy):
    """

    author @jilv220
    KC_BB Stra

    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 20.5
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.99

    use_custom_stoploss = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if (current_profit > 0.2):
            sl_new = 0.05
        elif (current_profit > 0.1):
            sl_new = 0.03
        elif (current_profit > 0.06):
            sl_new = 0.02
        elif (current_profit > 0.03):
            sl_new = 0.015

        return sl_new


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # True range
        dataframe['trange'] = ta.TRANGE(dataframe)

        # SMA
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_28'] = ta.SMA(dataframe, timeperiod=28)

        # KC 20 2
        dataframe['range_ma_20'] = ta.SMA(dataframe['trange'], 20)
        dataframe['kc_upperband_20_2'] = dataframe['sma_20'] + dataframe['range_ma_20'] * 2
        dataframe['kc_lowerband_20_2'] = dataframe['sma_20'] - dataframe['range_ma_20'] * 2

        # KC 28 1
        dataframe['range_ma_28'] = ta.SMA(dataframe['trange'], 28)
        dataframe['kc_upperband_28_1'] = dataframe['sma_28'] + dataframe['range_ma_28']
        dataframe['kc_lowerband_28_1'] = dataframe['sma_28'] - dataframe['range_ma_28']

        # BB 20 2
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['kc_bb_delta'] =  ( dataframe['kc_lowerband_20_2'] - dataframe['bb_lowerband2'] ) / dataframe['bb_lowerband2'] * 100

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()

        # fisher
        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # EMA
        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)

        # Williams R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[

                (dataframe['kc_lowerband_28_1'] < dataframe['bb_upperband2']) &
                (dataframe['kc_lowerband_28_1'] > dataframe['bb_lowerband2']) &
                (dataframe['kc_bb_delta'] < 1.5) &
                (dataframe['kc_bb_delta'] > 0.928) &
                (dataframe['r_14'] < -80) &
                (dataframe['r_14'] > -90) &
                (dataframe['ha_closedelta'] > dataframe['ha_close'] * 0.008)

            ,'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[

                (dataframe['fisher'] > 0.39075) &
                (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
                (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
                (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
                (dataframe['ema_fast'] > dataframe['ha_close']) &
                (dataframe['ha_close'] * 0.99754 > dataframe['bb_middleband2']) &
                (dataframe['volume'] > 0)

            ,'sell'] = 1

        return dataframe

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from âˆ’100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100
