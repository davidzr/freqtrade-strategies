# for live trailing_stop = False and use_custom_stoploss = True
# for backtest trailing_stop = True and use_custom_stoploss = False

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

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 8,
    "ewo_high": 2.403,
    "ewo_high_2": -5.585,
    "ewo_low": -14.378,
    "lookback_candles": 3,
    "low_offset": 0.984,
    "low_offset_2": 0.942,
    "profit_threshold": 1.008,
    "rsi_buy": 72
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 16,
    "high_offset": 1.084,
    "high_offset_2": 1.401,
    "pHSL": -0.15,
    "pPF_1": 0.016,
    "pPF_2": 0.024,
    "pSL_1": 0.014,
    "pSL_2": 0.022
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


class NASOSv4(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        # "0": 0.283,
        # "40": 0.086,
        # "99": 0.036,
        "0": 10
    }

    # Stoploss:
    stoploss = -0.15

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        2, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        2, 25, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=False)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=False)
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    lookback_candles = IntParameter(
        1, 24, default=buy_params['lookback_candles'], space='buy', optimize=True)

    profit_threshold = DecimalParameter(1.0, 1.03,
                                        default=buy_params['profit_threshold'], space='buy', optimize=True)

    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=False)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=False)

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=False)

    rsi_buy = IntParameter(50, 100, default=buy_params['rsi_buy'], space='buy', optimize=False)

    # trailing stoploss hyperopt parameters
    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.15, decimals=3,
                            space='sell', optimize=False, load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3,
                             space='sell', optimize=False, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.014, decimals=3,
                             space='sell', optimize=False, load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.024, decimals=3,
                             space='sell', optimize=False, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.022, decimals=3,
                             space='sell', optimize=False, load=True)

    # Trailing stop:
    trailing_stop = True
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
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = False

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }

    # Custom Trailing Stoploss by Perkmeister

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1)*(SL_2 - SL_1)/(PF_2 - PF_1))
        else:
            sl_profit = HSL

        # if current_profit < 0.001 and current_time - timedelta(minutes=600) > trade.open_date_utc:
        #     return -0.005

        return stoploss_from_open(sl_profit, current_profit)

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
        informative_pairs = [(pair, '1h') for pair in pairs]
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
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                # don't buy if there isn't 3% profit to be made
                (dataframe['close_1h'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < 35) &
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
                (dataframe['rsi_fast'] < 35) &
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
                (dataframe['rsi_fast'] < 35) &
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
