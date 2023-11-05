# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'

import technical.indicators as ftt
from technical.util import resample_to_interval, resampled_merge

from functools import reduce
from datetime import datetime, timedelta


# ObeliskIM v1.1 - 2021-03-24
#
# by Obelisk 
# https://twitter.com/brookmiles
#
# I watched a youtube video on Ichimoku and then wrote this last night at 3am while I couldn't sleep.
# Run at your own risk.
#
# Improvements and suggestions welcome!
#
# Potential areas of improvement:
# - reduce buying small peaks during sideways ranges
# - additional entry trigger(s) for strong up trends / breakouts if tk cross happened in/below cloud

class ObeliskIM_v1_1(IStrategy):

    # Optimal timeframe for the strategy
    timeframe = '5m'

    startup_candle_count = 288 # one day @ 5m
    process_only_new_candles = True

    minimal_roi = {
        "0": 5,
    }

    # Stoploss:
    stoploss = -0.04

    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(0,0,0,0.2)',
            },
            # plot senkou_b, too. Not only the area to it.
            'senkou_b': {
                'color': 'red',
            },
            'tenkan_sen': { 'color': 'orange' },
            'kijun_sen': { 'color': 'blue' },
        },
        'subplots': {
            "Ichimoku": {
                'cloud_green': {'color': 'green'},
                'cloud_red': {'color': 'red'},

                'cloud_green_strong': {'color': 'green'},
                'cloud_red_strong': {'color': 'red'},

                'tk_cross_up': {'color': 'blue'},
            },
            "RSI": {
                'rsi': {'color': 'blue'},
            },
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)


        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        ichimoku = ftt.ichimoku(dataframe, 
            conversion_line_period=20, 
            base_line_periods=60,
            laggin_span=120, 
            displacement=30
            )

        # cross indicators
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']

        # cloud, green a > b, red a < b
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        # dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        # dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        dataframe['cloud_red'] = ichimoku['cloud_red'] * -1

        dataframe['cloud_green_strong'] = (
                dataframe['cloud_green'] & 
                (dataframe['tenkan_sen'] > dataframe['kijun_sen']) &
                (dataframe['kijun_sen'] > dataframe['senkou_a'])
            ).astype('int') * 2

        dataframe['cloud_red_strong'] = (
                dataframe['cloud_red'] & 
                (dataframe['tenkan_sen'] < dataframe['kijun_sen']) &
                (dataframe['kijun_sen'] < dataframe['senkou_b'])
            ).astype('int') * -2

        dataframe.loc[
            qtpylib.crossed_above(dataframe['tenkan_sen'], dataframe['kijun_sen']),
            'tk_cross_up'] = 3
        dataframe['tk_cross_up'].fillna(method='ffill', inplace=True, limit=2)
        dataframe['tk_cross_up'].fillna(value=0, inplace=True)

        # dataframe['rsi_ok'] = (dataframe['rsi'] < 75).astype('int')
        dataframe['ema35_ok'] = (
            (dataframe['ema3'] > dataframe['ema5']) &
            (dataframe['ema5'] > dataframe['ema10'])
            ).astype('int')

        dataframe['spike'] = (
            (dataframe['close'] > (dataframe['close'].shift(3) * (1 - self.stoploss * 0.9)))
            ).astype('int')

        dataframe['recent_high'] = dataframe['high'].rolling(12).max()

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # params = self.buy_params

        conditions = []
        # conditions.append(dataframe['volume'] > 0)

        conditions.append( dataframe['cloud_green_strong'] > 0 )
        # conditions.append( dataframe['ema3'] > dataframe['ema5'] )
        conditions.append( dataframe['tk_cross_up'] > 0 )
        # conditions.append( dataframe['rsi_ok'] > 0 )
        conditions.append( dataframe['ema35_ok'] > 0 )
        conditions.append( dataframe['close'] > dataframe['close'].shift() )
        conditions.append( dataframe['close'] > dataframe['recent_high'].shift() ) # remove me for a "good time"
        conditions.append( dataframe['spike'] < 1 )

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # params = self.sell_params

        conditions = []
        # conditions.append(dataframe['volume'] > 0)

        conditions.append( 
            qtpylib.crossed_below(dataframe['tenkan_sen'], dataframe['kijun_sen']) | 
            qtpylib.crossed_below(dataframe['close'], dataframe['kijun_sen'])
            )

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1

        return dataframe
