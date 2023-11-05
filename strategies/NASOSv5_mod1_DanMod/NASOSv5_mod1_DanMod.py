import datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
import technical.indicators as ftt
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
from freqtrade.strategy.interface import IStrategy
from functools import reduce
from logging import FATAL
from pandas import DataFrame
from technical.util import resample_to_interval, resampled_merge
from typing import Dict, List

logger = logging.getLogger(__name__)

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif
    
def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def top_percent_change(dataframe: DataFrame, length: int) -> float:
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

class NASOSv5_mod1_DanMod(IStrategy):
    INTERFACE_VERSION = 2

    buy_params = {
        "base_nb_candles_buy": 20,
        "ewo_high": 4.299,
        "ewo_high_2": 8.492,
        "ewo_low": -8.476,
        "low_offset": 0.984,
        "low_offset_2": 0.901,
        "lookback_candles": 7,
        "profit_threshold": 1.036,
        "rsi_buy": 80,
        "rsi_fast_buy": 27,
    }

    sell_params = {
        "base_nb_candles_sell": 20,
        "high_offset": 1.01,
        "high_offset_2": 1.142,
    }

    minimal_roi = {
        "0": 0.4
    }

    stoploss = -0.3

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    base_nb_candles_buy = IntParameter(
        2, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        2, 25, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)

    fast_ewo = 50
    slow_ewo = 200

    lookback_candles = IntParameter(
        1, 36, default=buy_params['lookback_candles'], space='buy', optimize=True)

    profit_threshold = DecimalParameter(0.99, 1.05,
                                        default=buy_params['profit_threshold'], space='buy', optimize=True)

    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=True)

    rsi_buy = IntParameter(10, 80, default=buy_params['rsi_buy'], space='buy', optimize=True)
    rsi_fast_buy = IntParameter(
        10, 50, default=buy_params['rsi_fast_buy'], space='buy', optimize=True)

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    timeframe = '5m'
    inf_15m = '15m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = False

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
        'subplots': {
            'rsi': {
                'rsi': {'color': 'orange'},
                'rsi_fast': {'color': 'red'},
                'rsi_slow': {'color': 'green'},
            },
            'ewo': {
                'EWO': {'color': 'orange'}
            },
        }
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }

    protections = [
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,
            "trade_limit": 1,
            "stop_duration": 60,
            "required_profit": -0.05
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 24,
            "trade_limit": 1,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.2
        },
    ]

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if (current_profit > 0.3):
            return 0.05
        elif (current_profit > 0.1):
            return 0.03
        elif (current_profit > 0.06):
            return 0.02
        elif (current_profit > 0.04):
            return 0.01
        elif (current_profit > 0.025):
            return 0.005
        elif (current_profit > 0.018):
            return 0.005

        return 0.15

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951):  # *1.2
                    return False

        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '15m') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        
        return informative_1h

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)
        
        return informative_15m

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        dataframe['tcp_percent_4'] = top_percent_change(dataframe , 4)
        
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=5)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=25)

        dataframe['pct_change'] = dataframe['close'].pct_change(periods=8)
        dataframe['pct_change_int'] = ((dataframe['pct_change'] > 0.15).astype(int) | (dataframe['pct_change'] < -0.15).astype(int))

        dataframe['pct_change_short'] = dataframe['close'].pct_change(periods=8)
        dataframe['pct_change_int_short'] = ((dataframe['pct_change_short'] > 0.08).astype(int) | (dataframe['pct_change_short'] < -0.08).astype(int))

        dataframe['ispumping'] = (
         (dataframe['pct_change_int'].rolling(20).sum() >= 0.4)
        ).astype('int')

        dataframe['islongpumping'] = (
         (dataframe['pct_change_int'].rolling(30).sum() >= 0.48)
        ).astype('int')

        dataframe['isshortpumping'] = (
         (dataframe['pct_change_int_short'].rolling(10).sum() >= 0.10)
        ).astype('int')

        dataframe['recentispumping'] = (dataframe['ispumping'].rolling(300).max() > 0) | (dataframe['islongpumping'].rolling(300).max() > 0)# | (dataframe['isshortpumping'].rolling(300).max() > 0)
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)

        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                (dataframe['close_15m'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < dataframe['vwap_low']) &
                (dataframe['tcp_percent_4'] > 0.04) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['rsi'] < 25)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            ((dataframe['close'] > dataframe['sma_9']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
             )
            |
            (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe