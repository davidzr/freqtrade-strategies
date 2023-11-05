# --- Do not remove these libs ---
# --- Do not remove these libs ---
from logging import FATAL
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

# @Rallipanos
# @pluxury
# with help from @stash86 and @Perkmeister

# Buy hyperspace params:
buy_params = {
    "low_offset": 0.981,
    "base_nb_candles_buy": 8,  # value loaded from strategy
    "ewo_high": 3.553,  # value loaded from strategy
    "ewo_high_2": -5.585,  # value loaded from strategy
    "ewo_low": -14.378,  # value loaded from strategy
    "lookback_candles": 32,  # value loaded from strategy
    "low_offset_2": 0.942,  # value loaded from strategy
    "profit_threshold": 1.037,  # value loaded from strategy
    "rsi_buy": 78,  # value loaded from strategy
    "rsi_fast_buy": 37,  # value loaded from strategy
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 16,  # value loaded from strategy
    "high_offset": 1.097,  # value loaded from strategy
    "high_offset_2": 1.472,  # value loaded from strategy
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


class NASOSv5(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        # "0": 0.283,
        # "40": 0.086,
        # "99": 0.036,
        "360": 0
    }

    # Stoploss:
    stoploss = -0.15

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        2, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(
        2, 25, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=False)
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=False)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    lookback_candles = IntParameter(
        1, 36, default=buy_params['lookback_candles'], space='buy', optimize=False)

    profit_threshold = DecimalParameter(0.99, 1.05,
                                        default=buy_params['profit_threshold'], space='buy', optimize=False)

    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=False)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=False)

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=False)

    rsi_buy = IntParameter(10, 80, default=buy_params['rsi_buy'], space='buy', optimize=False)
    rsi_fast_buy = IntParameter(
        10, 50, default=buy_params['rsi_fast_buy'], space='buy', optimize=False)

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_15m = '15m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = True

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

        # slippage
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
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)
        # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_15m

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # informative_1h = self.informative_1h_indicators(dataframe, metadata)
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                # don't buy if there isn't 3% profit to be made
                (dataframe['close_15m'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
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
