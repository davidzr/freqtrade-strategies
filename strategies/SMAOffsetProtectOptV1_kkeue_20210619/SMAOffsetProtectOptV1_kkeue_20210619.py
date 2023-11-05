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

######################################## Warning ########################################
# You won't get a lot of benefits by simply changing to this strategy                   #
# with the HyperOpt values changed.                                                     #
#                                                                                       #
# You should test it closely, trying backtesting and dry running, and we recommend      #
# customizing the terms of sale and purchase as well.                                   #
#                                                                                       #
# You should always be careful in real trading!                                         #
#########################################################################################



def EWO(dataframe, ema_length=5, ema2_length=35):
    #df = dataframe.copy()
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / dataframe['close'] * 100
    return emadif


class SMAOffsetProtectOptV1_kkeue_20210619(IStrategy):
    # Modified Buy / Sell params - 20210619
    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 16,
        "ewo_high": 5.672,
        "ewo_low": -19.931,
        "low_offset": 0.973,
        "rsi_buy": 59,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 20,
        "high_offset": 1.010,
    }
    INTERFACE_VERSION = 2

    # Modified ROI - 20210620
    # ROI table:
    minimal_roi = {
        "0": 0.028,
        "10": 0.018,
        "30": 0.010,
        "40": 0.005
    }

    # Stoploss:
    stoploss = -0.5

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)


    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count: int = 30

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    use_custom_stoploss = False

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
        
        
class SMAOffsetProtectOptV1_1(SMAOffsetProtectOptV1_kkeue_20210619):
    #Epoch details:

    #271/512:    468 trades. 453/0/15 Wins/Draws/Losses. Avg profit   1.11%. Median profit   1.14%. Total profit  129.62363315 BUSD ( 103.70%). Avg duration 2:07:00 min. Objective: -50137.47104


    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 5,
        "ewo_high": 3.944,
        "ewo_low": -12.07,
        "low_offset": 0.987,
        "rsi_buy": 69,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 53,
        "high_offset": 1.044,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.028,
        "10": 0.018,
        "30": 0.01,
        "40": 0.005
    }

    # Stoploss:
    stoploss = -0.5  # value loaded from strategy

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = 0.001  # value loaded from strategy
    trailing_stop_positive_offset = 0.01  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy