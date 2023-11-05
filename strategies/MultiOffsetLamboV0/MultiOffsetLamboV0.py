# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt

# - Credits -
# tirail: SMAOffset idea
# rextea: EWO idea
# Lambo


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class MultiOffsetLamboV0(IStrategy):
    INTERFACE_VERSION = 2

    # Hyperopt Result

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 16,
        "ewo_high": 5.638,
        "ewo_low": -19.993
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 49
    }

    # ROI table:
    minimal_roi = {
        "0": 0.01
    }

    # Stoploss:
    stoploss = -0.50

    # Offset
    base_nb_candles_buy = IntParameter(
        5, 80, default=20, load=True, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=20, load=True, space='sell', optimize=True)
    low_offset_sma = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_sma = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_ema = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_ema = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_trima = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_trima = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_t3 = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_t3 = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_kama = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_kama = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)

    # Protection
    ewo_low = DecimalParameter(
        -20.0, -8.0, default=-20.0, load=True, space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=6.0, load=True, space='buy', optimize=True)
    fast_ewo = IntParameter(
        10, 50, default=50, load=True, space='buy', optimize=False)
    slow_ewo = IntParameter(
        100, 200, default=200, load=True, space='buy', optimize=False)

    # MA list
    ma_types = ['sma', 'ema', 'trima', 't3', 'kama']
    ma_map = {
        'sma': {
            'low_offset': low_offset_sma.value,
            'high_offset': high_offset_sma.value,
            'calculate': ta.SMA
        },
        'ema': {
            'low_offset': low_offset_ema.value,
            'high_offset': high_offset_ema.value,
            'calculate': ta.EMA
        },
        'trima': {
            'low_offset': low_offset_trima.value,
            'high_offset': high_offset_trima.value,
            'calculate': ta.TRIMA
        },
        't3': {
            'low_offset': low_offset_t3.value,
            'high_offset': high_offset_t3.value,
            'calculate': ta.T3
        },
        'kama': {
            'low_offset': low_offset_kama.value,
            'high_offset': high_offset_kama.value,
            'calculate': ta.KAMA
        }
    }

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    use_sell_signal = True
    sell_profit_only = False

    process_only_new_candles = True
    startup_candle_count = 30

    plot_config = {
        'main_plot': {
            'ma_offset_buy': {'color': 'orange'},
            'ma_offset_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Offset
        for i in self.ma_types:
            dataframe[f'{i}_offset_buy'] = self.ma_map[f'{i}']['calculate'](
                dataframe, self.base_nb_candles_buy.value) * \
                self.ma_map[f'{i}']['low_offset']
            dataframe[f'{i}_offset_sell'] = self.ma_map[f'{i}']['calculate'](
                dataframe, self.base_nb_candles_sell.value) * \
                self.ma_map[f'{i}']['high_offset']

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        for i in self.ma_types:
            conditions.append(
                (dataframe['close'] < dataframe[f'{i}_offset_buy']) &
                (
                    (dataframe['EWO'] < self.ewo_low.value) |
                    (dataframe['EWO'] > self.ewo_high.value)
                ) &
                (dataframe['volume'] > 0)
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        for i in self.ma_types:
            conditions.append(
                (
                    (dataframe['close'] > dataframe[f'{i}_offset_sell']) &
                    (dataframe['volume'] > 0)
                )
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
