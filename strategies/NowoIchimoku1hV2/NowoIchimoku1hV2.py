# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import math
from typing import Callable

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter

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


class NowoIchimoku1hV2(IStrategy):
    # Optimal timeframe for the strategy
    timeframe = '1h'

    startup_candle_count = 100

    use_sell_signal = False

    use_custom_stoploss = True

    trailing_stop = True

    minimal_roi = {
        "0": 0.427,
        "448": 0.202,
        "1123": 0.089,
        "2355": 0
    }

    stoploss = -0.345

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
            'upper': {
                'color': 'blue'
            }
        },
        'subplots': {
            'Buy Allowed': {
                'buy_allowed': {
                    'color': 'blue'
                },
            },
            'Should Buy': {
                'should_buy': {
                    'color': 'green'
                },
            },
            'Is Cloud Green': {
                'is_cloud_green': {
                    'color': 'black'
                }
            }
        }
    }

    srsi_k_min_profit = DecimalParameter(0.01, 0.99, decimals=3, default=0.036, space="sell")
    above_upper_min_profit = DecimalParameter(0.001, 0.5, decimals=3, default=0.011, space="sell")
    limit_factor = DecimalParameter(0.5, 5, decimals=3, default=1.918, space="sell")
    lower_cloud_factor = DecimalParameter(0.5, 1.5, decimals=3, default=0.971, space="sell")
    close_above_shifted_upper_cloud = DecimalParameter(0.5, 2, decimals=3, default=0.603, space="buy")

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()

        if (last_candle is not None) & (previous_candle is not None):
            # In dry/live runs trade open date will not match candle open date therefore it must be
            # rounded.
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            # Look up trade candle.
            trade_candle = dataframe.loc[dataframe['date'] == trade_date]

            # trade_candle may be empty for trades that just opened as it is still incomplete.
            if not trade_candle.empty:
                trade_candle = trade_candle.squeeze()

                if (last_candle['srsi_k'] > 80) & (current_profit > self.srsi_k_min_profit.value):
                    return -0.0001

                if (previous_candle['close'] < previous_candle['upper']) & (current_rate > last_candle['upper']) & (
                        current_profit > self.above_upper_min_profit.value):
                    return -0.0001

                limit = trade.open_rate + ((trade.open_rate - trade_candle['shifted_lower_cloud']) * self.limit_factor.value)

                if current_rate > limit:
                    return -0.0001

                if current_rate < (trade_candle['shifted_lower_cloud'] * self.lower_cloud_factor.value):
                    return -0.0001

        return -0.99

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

        close_above_shifted_upper_cloud = df['close'] > df['shifted_upper_cloud'] * self.close_above_shifted_upper_cloud.value
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

        df['buy'] = False
        df['buy_allowed'] = True
        for row in df.itertuples():
            if row.Index > 100:
                df.loc[row.Index, 'buy_allowed'] = df.at[row.Index - 1, 'buy_allowed']

                if df.at[row.Index - 1, 'buy']:
                    df.loc[row.Index, 'buy_allowed'] = False

                if not df.at[row.Index, 'is_cloud_green']:
                    df.loc[row.Index, 'buy_allowed'] = True

                df.loc[row.Index, 'buy'] = df.at[row.Index, 'buy_allowed'] & df.at[row.Index, 'should_buy']

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['sell'] = 0
        return df
