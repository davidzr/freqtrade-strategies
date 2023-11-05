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
from freqtrade.strategy import DecimalParameter, IntParameter
from freqtrade.exchange import timeframe_to_minutes

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


def merge_informative_pair(dataframe: pd.DataFrame, informative: pd.DataFrame,
                           timeframe: str, timeframe_inf: str, ffill: bool = True) -> pd.DataFrame:
    """
   Correctly merge informative samples to the original dataframe, avoiding lookahead bias.

   Since dates are candle open dates, merging a 15m candle that starts at 15:00, and a
   1h candle that starts at 15:00 will result in all candles to know the close at 16:00
   which they should not know.

   Moves the date of the informative pair by 1 time interval forward.
   This way, the 14:00 1h candle is merged to 15:00 15m candle, since the 14:00 1h candle is the
   last candle that's closed at 15:00, 15:15, 15:30 or 15:45.

   Assuming inf_tf = '1d' - then the resulting columns will be:
   date_1d, open_1d, high_1d, low_1d, close_1d, rsi_1d

   :param dataframe: Original dataframe
   :param informative: Informative pair, most likely loaded via dp.get_pair_dataframe
   :param timeframe: Timeframe of the original pair sample.
   :param timeframe_inf: Timeframe of the informative pair sample.
   :param ffill: Forwardfill missing values - optional but usually required
   :return: Merged dataframe
   :raise: ValueError if the secondary timeframe is shorter than the dataframe timeframe
   """

    minutes_inf = timeframe_to_minutes(timeframe_inf)
    minutes = timeframe_to_minutes(timeframe)
    if minutes == minutes_inf:
        # No need to forwardshift if the timeframes are identical
        informative['date_merge'] = informative["date"]
    elif minutes < minutes_inf:
        # Subtract "small" timeframe so merging is not delayed by 1 small candle
        # Detailed explanation in https://github.com/freqtrade/freqtrade/issues/4073
        informative['date_merge'] = (
                informative["date"] + pd.to_timedelta(minutes_inf, 'm') - pd.to_timedelta(minutes, 'm')
        )
    else:
        raise ValueError("Tried to merge a faster timeframe to a slower timeframe."
                         "This would create new rows, and can throw off your regular indicators.")

    # Rename columns to be unique
    informative.columns = [f"{col}_{timeframe_inf}" for col in informative.columns]

    # Combine the 2 dataframes
    # all indicators on the informative sample MUST be calculated before this point
    dataframe = pd.merge(dataframe, informative, left_on='date',
                         right_on=f'date_merge_{timeframe_inf}', how='left')
    dataframe = dataframe.drop(f'date_merge_{timeframe_inf}', axis=1)

    if ffill:
        dataframe = dataframe.ffill()

    return dataframe


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


class NowoIchimoku5mV2(IStrategy):
    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    time_factor = int(60 / 5)

    startup_candle_count = int(100 * time_factor)

    use_sell_signal = False

    use_custom_stoploss = True

    trailing_stop = True

    minimal_roi = {
        "0": 999
    }

    stoploss = -0.293

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

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, df_5m: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."

        # Get the informative pair
        df_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        df_1h['upper'] = bollinger_bands(df_1h['close'], moving_average='hma', length=20, mult=2.5)['upper']

        ichi_1h = indicators.ichimoku(df_1h)

        df_1h['conversion_line'] = ichi_1h['tenkan_sen']
        df_1h['base_line'] = ichi_1h['kijun_sen']

        df_1h['lead_1'] = ichi_1h['leading_senkou_span_a']
        df_1h['lead_2'] = ichi_1h['leading_senkou_span_b']

        df_1h['is_cloud_green'] = ichi_1h['cloud_green']

        df_1h['upper_cloud'] = df_1h['lead_1'].where(df_1h['lead_1'] > df_1h['lead_2'], df_1h['lead_2'])
        df_1h['lower_cloud'] = df_1h['lead_1'].where(df_1h['lead_1'] < df_1h['lead_2'], df_1h['lead_2'])

        df_1h['shifted_upper_cloud'] = df_1h['upper_cloud'].shift(25)
        df_1h['shifted_lower_cloud'] = df_1h['lower_cloud'].shift(25)

        smoothK = 3
        smoothD = 3
        lengthRSI = 14
        lengthStoch = 14

        df_1h['rsi'] = ta.RSI(df_1h, timeperiod=lengthRSI)

        stochrsi = (df_1h['rsi'] - df_1h['rsi'].rolling(lengthStoch).min()) / (
                df_1h['rsi'].rolling(lengthStoch).max() - df_1h['rsi'].rolling(lengthStoch).min())

        df_1h['srsi_k'] = stochrsi.rolling(smoothK).mean() * 100
        df_1h['srsi_d'] = df_1h['srsi_k'].rolling(smoothD).mean()

        df = merge_informative_pair(df_5m, df_1h, self.timeframe, self.informative_timeframe, ffill=True)
        # don't overwrite the base dataframe's OHLCV information
        skip_columns = [(s + "_" + self.informative_timeframe) for s in
                        ['date', 'open', 'high', 'low', 'close', 'volume']]
        df.rename(
            columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (not s in skip_columns) else s,
            inplace=True)

        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1 * self.time_factor].squeeze()
        previous_candle = dataframe.iloc[-2 * self.time_factor].squeeze()

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

                limit = trade.open_rate + (
                        (trade.open_rate - trade_candle['shifted_lower_cloud']) * self.limit_factor.value)

                if current_rate > limit:
                    return -0.0001

                if current_rate < (trade_candle['shifted_lower_cloud'] * self.lower_cloud_factor.value):
                    return -0.0001

        return -0.99

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.info(verbose=True)

        double_shifted_upper_cloud = df['upper_cloud'].shift(50 * self.time_factor)

        close_above_shifted_upper_cloud = df['close'] > df[
            'shifted_upper_cloud'] * self.close_above_shifted_upper_cloud.value
        close_above_shifted_lower_cloud = df['close'] > df['shifted_lower_cloud']
        close_above_double_shifted_upper_cloud = df['close'] > double_shifted_upper_cloud

        conversion_line_above_base_line = df['conversion_line'] > df['base_line']
        close_above_shifted_conversion_line = df['close'] > df['conversion_line'].shift(25 * self.time_factor)

        df['should_buy'] = (df['close'] > df['open']) & \
                           close_above_shifted_upper_cloud & \
                           close_above_shifted_lower_cloud & \
                           df['is_cloud_green'] & \
                           conversion_line_above_base_line & \
                           close_above_shifted_conversion_line & \
                           close_above_double_shifted_upper_cloud

        df['buy'] = False
        df['buy_allowed'] = True

        for i in range(1, len(df)):
            df.loc[i, 'buy_allowed'] = df.at[i - 1, 'buy_allowed']

            if df.at[i - 1, 'buy']:
                df.loc[i, 'buy_allowed'] = False

            if not df.at[i, 'is_cloud_green']:
                df.loc[i, 'buy_allowed'] = True

            df.loc[i, 'buy'] = df.at[i, 'buy_allowed'] & df.at[i, 'should_buy']

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['sell'] = 0
        return df
