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


SMA = 'SMA'
EMA = 'EMA'

# Buy hyperspace params:
#buy_params = {
#    "base_nb_candles_buy": 20,
#    "ewo_high": 6,
#    "fast_ewo": 50,
#    "slow_ewo": 200,
#    "low_offset": 0.958,
#    "buy_trigger": "EMA",
#    "ewo_high": 2.0,
#    "ewo_low": -16.062,
#    "rsi_buy": 51,
#}

buy_params = {
    "base_nb_candles_buy": 20,
    "ewo_high": 5.499,
    "ewo_low": -19.881,
    "low_offset": 0.975,
    "rsi_buy": 67,
    "buy_trigger": "EMA",  # value loaded from strategy
    "fast_ewo": 50,  # value loaded from strategy
    "slow_ewo": 200,  # value loaded from strategy
    "buy_trigger": "EMA",
}

# Sell hyperspace params:
#sell_params = {
#    "base_nb_candles_sell": 20,
#    "high_offset": 1.012,
#    "sell_trigger": "EMA",
#}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 24,
    "high_offset": 1.012,
    "sell_trigger": "EMA",
}



def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class SMAOffsetProtectOptV0(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.01
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
    buy_trigger = CategoricalParameter(
        [SMA, EMA], default=buy_params['buy_trigger'], space='buy', optimize=False)
    sell_trigger = CategoricalParameter(
        [SMA, EMA], default=sell_params['sell_trigger'], space='sell', optimize=False)

    # Protection
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    fast_ewo = IntParameter(
        10, 50, default=buy_params['fast_ewo'], space='buy', optimize=False)
    slow_ewo = IntParameter(
        100, 200, default=buy_params['slow_ewo'], space='buy', optimize=False)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)
    # slow_ema = IntParameter(
    #     10, 50, default=buy_params['fast_ewo'], space='buy', optimize=True)
    # fast_ema = IntParameter(
    #     100, 200, default=buy_params['slow_ewo'], space='buy', optimize=True)

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

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        # EMA
        informative_pairs['ema_50'] = ta.EMA(informative_pairs, timeperiod=50)
        informative_pairs['ema_100'] = ta.EMA(informative_pairs, timeperiod=100)
        informative_pairs['ema_200'] = ta.EMA(informative_pairs, timeperiod=200)
        # SMA
        informative_pairs['sma_200'] = ta.SMA(informative_pairs, timeperiod=200)
        informative_pairs['sma_200_dec'] = informative_pairs['sma_200'] < informative_pairs['sma_200'].shift(
            20)
        # RSI
        informative_pairs['rsi'] = ta.RSI(informative_pairs, timeperiod=14)

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # informative = self.get_informative_indicators(metadata)
        # dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
        #                                    ffill=True)

        # Calculate all base_nb_candles_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all base_nb_candles_buy values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)


        # ---------------- original code -------------------
        ##SMAOffset
        #if self.buy_trigger.value == 'EMA':
        #    dataframe['ma_buy'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_buy.value)
        #else:
        #    dataframe['ma_buy'] = ta.SMA(dataframe, timeperiod=self.base_nb_candles_buy.value)
        #
        #if self.sell_trigger.value == 'EMA':
        #    dataframe['ma_sell'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_sell.value)
        #else:
        #    dataframe['ma_sell'] = ta.SMA(dataframe, timeperiod=self.base_nb_candles_sell.value)
        #
        #dataframe['ma_offset_buy'] = dataframe['ma_buy'] * self.low_offset.value
        #dataframe['ma_offset_sell'] = dataframe['ma_sell'] * self.high_offset.value
        # ------------ end original code --------------------

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)
        
        
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

        # ---------------- original code -------------------
        #conditions.append(
        #    (
        #        (dataframe['close'] < dataframe['ma_offset_buy']) &
        #        (dataframe['EWO'] > self.ewo_high.value) &
        #        (dataframe['rsi'] < self.rsi_buy.value) &
        #        (dataframe['volume'] > 0)
        #    )
        #)

        #conditions.append(
        #    (
        #        (dataframe['close'] < dataframe['ma_offset_buy']) &
        #        (dataframe['EWO'] < self.ewo_low.value) &
        #        (dataframe['volume'] > 0)
        #    )
        #)
        # ------------ end original code --------------------

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


        # ---------------- original code -------------------
        #conditions.append(
        #    (
        #        (dataframe['close'] > dataframe['ma_offset_sell']) &
        #        (dataframe['volume'] > 0)
        #    )
        #)
        # ------------ end original code --------------------

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
