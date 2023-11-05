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

# Buy hyperspace params: orginal
# buy_params = {
#     "base_nb_candles_buy": 16,
#     "ewo_high": 5.638,
#     "ewo_low": -19.993,
#     "low_offset": 0.978,
#     "rsi_buy": 61,
#    "fast_ewo": 50,  # value loaded from strategy
#    "slow_ewo": 200,  # value loaded from strategy
# }
# Buy hyperspace params: from v0
buy_params = {
    "base_nb_candles_buy": 20,
    "ewo_high": 5.499,
    "ewo_low": -19.881,
    "low_offset": 0.975,
    "rsi_buy": 67,
    "fast_ewo": 50,  # value loaded from strategy
    "slow_ewo": 200,  # value loaded from strategy

}
# Sell hyperspace params:orginal
# sell_params = {
#     "base_nb_candles_sell": 49,
#     "high_offset": 1.006,
# }
# Sell hyperspace params:  from v0
sell_params = {
    "base_nb_candles_sell": 24,
    "high_offset": 1.012,
}

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class SMAOffsetProtectOpt(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.20,
        "38": 0.074,
        "78": 0.025,
        "194": 0
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
    fast_ewo = IntParameter(
        10, 50, default=buy_params['fast_ewo'], space='buy', optimize=False)
    slow_ewo = IntParameter(
        100, 200, default=buy_params['slow_ewo'], space='buy', optimize=False)
    # fast_ewo = 50
    # slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)


    # Trailing stop:
    # trailing_stop = False
    # trailing_stop_positive = 0.001
    # trailing_stop_positive_offset = 0.01
    # trailing_only_offset_is_reached = True

    # Sell signal
    # use_sell_signal = True
    # sell_profit_only = False
    # sell_profit_offset = 0.01
    # ignore_roi_if_buy_signal = True

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 200

    # plot_config = {
    #     'main_plot': {
    #         'ma_buy': {'color': 'orange'},
    #         'ma_sell': {'color': 'orange'},
    #     },
    # }

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
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe['ma_buy'] = (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)

        conditions.append(
            (
                # (dataframe['close'].shift(1) < dataframe['ma_buy']) &
                # (dataframe['low'] < dataframe['ma_buy']) &
                # (dataframe['close'] > dataframe['ma_buy']) &
                # (qtpylib.crossed_above(dataframe['close'], dataframe['ma_buy'])) &
                (dataframe['close'] < dataframe['ma_buy']) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                # (dataframe['close'].shift(1) < dataframe['ma_buy']) &
                # (dataframe['low'] < dataframe['ma_buy']) &
                # (dataframe['close'] > dataframe['ma_buy']) &
                # (qtpylib.crossed_above(dataframe['close'], dataframe['ma_buy'])) &
                (dataframe['close'] < dataframe['ma_buy']) &
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
        dataframe['ma_sell']= (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)
        conditions.append(
            (
                #(dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (qtpylib.crossed_below(dataframe['close'], dataframe['ma_sell'])) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
