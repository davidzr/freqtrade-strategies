# This version of the strategy is broken!

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import math
from typing import Callable

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IntParameter
from pandas import Series
from numpy.typing import ArrayLike
from datetime import datetime, timedelta
import technical.indicators as indicators
from freqtrade.exchange import timeframe_to_prev_date
from finta import TA


def wma(series: Series, length: int) -> Series:
    norm = 0
    sum = 0

    for i in range(1, length - 1):
        weight = (length - i) * length
        norm = norm + weight
        sum = sum + series.shift(i) * weight

    return sum / norm


def hma(series: Series, length: int) -> Series:
    h = 2 * wma(series, math.floor(length / 2)) - wma(series, length)
    hma = wma(h, math.floor(math.sqrt(length)))
    return hma


def bollinger_bands(series: Series, moving_average='sma', length=20, mult=2.0) -> DataFrame:
    basis = None
    if moving_average == 'sma':
        basis = ta.SMA(series, length)
    elif moving_average == 'hma':
        basis = hma(series, length)
    else:
        raise Exception("moving_average has to be sma or hma")

    dev = mult * ta.STDDEV(series, length)

    return DataFrame({
        'upper': basis + dev
    })


class NowoIchimoku1hV1(IStrategy):
    # Optimal timeframe for the strategy
    timeframe = '1h'

    startup_candle_count = 100

    use_sell_signal = False

    use_custom_stoploss = True

    minimal_roi = {
        "0": 999
    }

    stoploss = -0.08

    plot_config = {
        'main_plot': {
            # 'lead_1': {
            #     'color': 'green',
            #     'fill_to': 'lead_2',
            #     'fill_label': 'Ichimoku Cloud',
            #     'fill_color': 'rgba(0,0,0,0.2)',
            # },
            # 'lead_2': {
            #     'color': 'red',
            # },
            # 'conversion_line': {'color': 'blue'},
            # 'base_line': {'color': 'orange'},
        },
    }

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # In dry/live runs trade open date will not match candle open date therefore it must be
        # rounded.
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # Look up trade candle.
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        # trade_candle may be empty for trades that just opened as it is still incomplete.
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()

            if last_candle['srsi_k'] > 80 & current_profit > 1.1:
                return 'srsi_k above 80 with profit above 10%'

            if current_rate > last_candle['upper'] & current_profit > 1.01:
                return 'current rate above upper band with profit above 1%'

            limit = trade_candle['close'] + ((trade_candle['close'] - trade_candle['shifted_lower_cloud']) * 2)

            if current_rate > limit:
                return 'current rate above limit'

            if current_rate < trade_candle['shifted_lower_cloud']:
                return 'current rate below stop'

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        return 1

    # def informative_pairs(self):
    #     return []

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['upper'] = bollinger_bands(df['close'], moving_average='hma', length=20, mult=2.5)['upper']

        ichi = indicators.ichimoku(df)

        df['conversion_line'] = ichi['tenkan_sen']
        df['base_line'] = ichi['kijun_sen']

        df['lead_1'] = ichi['leading_senkou_span_a']
        df['lead_2'] = ichi['leading_senkou_span_b']

        df['cloud_green'] = ichi['cloud_green']

        df['upper_cloud'] = df['lead_1'].where(df['lead_1'] > df['lead_2'], df['lead_2'])
        df['lower_cloud'] = df['lead_1'].where(df['lead_1'] < df['lead_2'], df['lead_2'])

        df['shifted_upper_cloud'] = df['upper_cloud'].shift(25)
        df['shifted_lower_cloud'] = df['lower_cloud'].shift(25)

        smoothK = 3
        smoothD = 3
        lengthRSI = 14
        lengthStoch = 14

        df['rsi'] = ta.RSI(df, timeperiod=lengthRSI)

        stochrsi = (df['rsi'] - df['rsi'].rolling(lengthStoch).min()) / (
                df['rsi'].rolling(lengthStoch).max() - df['rsi'].rolling(lengthStoch).min())

        df['srsi_k'] = stochrsi.rolling(smoothK).mean() * 100
        df['srsi_d'] = df['srsi_k'].rolling(smoothD).mean()

        # df['srsi_top'] = 80
        # df['srsi_bottom'] = 20

        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['is_cloud_green'] = df['lead_1'] > df['lead_2']

        double_shifted_upper_cloud = df['upper_cloud'].shift(50)

        close_above_shifted_upper_cloud = df['close'] > df['shifted_upper_cloud'] * 1.04
        close_above_shifted_lower_cloud = df['close'] > df['shifted_lower_cloud']
        close_above_double_shifted_upper_cloud = df['close'] > double_shifted_upper_cloud

        conversion_line_above_base_line = df['conversion_line'] > df['base_line']
        close_above_shifted_conversion_line = df['close'] > df['conversion_line'].shift(25)

        df['should_buy'] = (df['close'] > df['open']) & \
                           close_above_shifted_upper_cloud & \
                           close_above_shifted_lower_cloud & \
                           df['is_cloud_green'] & \
                           conversion_line_above_base_line & \
                           close_above_shifted_conversion_line & \
                           close_above_double_shifted_upper_cloud

        df['buy_allowed'] = -df['is_cloud_green']
        for i in range(101, len(df)):
            df.loc[i, 'buy_allowed'] = df.loc[i - 1, 'buy_allowed']

            if df.loc[i - 1, 'buy']:
                df.loc[i, 'buy_allowed'] = False

            elif not df.loc[i, 'is_cloud_green']:
                df[i, 'buy_allowed'] = True

        df.loc[
            (df['buy_allowed'] & df['should_buy'])
            , 'buy'] = 1

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['sell'] = 0
        return df
