from typing import Optional
from functools import reduce
from typing import List

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter, IntParameter, BooleanParameter
from pandas import DataFrame, Series
from datetime import datetime, timedelta, timezone


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


class fahmibah(IStrategy):
    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """
    
    # hypered params
    buy_params = {
        "clucha_enabled": True,
        "bbdelta_close": 0.01889,
        "bbdelta_tail": 0.72235,
        "close_bblower": 0.0127,
        "closedelta_close": 0.00916,
        "rocr_1h": 0.79492,
        "lambo1_enabled": True,
        "lambo1_ema_14_factor": 1.054,
        "lambo1_rsi_4_limit": 18,
        "lambo1_rsi_14_limit": 39,
    }

    # Sell hyperspace params:
    sell_params = {
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.10,
        "pPF_1": 0.02,
        "pPF_2": 0.03,
        "pSL_1": 0.015,
        "pSL_2": 0.025,

        # sell signal params
        'sell_fisher': 0.39075,
        'sell_bbmiddle_close': 0.99754
    }

    # ROI table:
    minimal_roi = {
        "0": 0.033,
        "10": 0.023,
        "40": 0.01,
    }

    # Stoploss:
    stoploss = -0.10  # use custom stoploss

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """

    timeframe = '5m'

    # Make sure these match or are not overridden in config
    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Custom stoploss
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 168

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'emergencysell': 'limit',
        'forcebuy': "limit",
        'forcesell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # buy params ClucHA
    clucha_enabled = BooleanParameter(default=buy_params['clucha_enabled'], space='buy', optimize=False)
    rocr_1h = DecimalParameter(0.5, 1.0, default=0.54904, space='buy', decimals=5, optimize=False)
    bbdelta_close = DecimalParameter(0.0005, 0.02, default=0.01965, space='buy', decimals=5, optimize=False)
    closedelta_close = DecimalParameter(0.0005, 0.02, default=0.00556, space='buy', decimals=5, optimize=False)
    bbdelta_tail = DecimalParameter(0.7, 1.0, default=0.95089, space='buy', decimals=5, optimize=False)
    close_bblower = DecimalParameter(0.0005, 0.02, default=0.00799, space='buy', decimals=5, optimize=False)

    # buy params lambo1
    lambo1_enabled = BooleanParameter(default=buy_params['lambo1_enabled'], space='buy', optimize=False)
    lambo1_ema_14_factor = DecimalParameter(0.5, 2.0, default=1.054, space='buy', decimals=3, optimize=True)
    lambo1_rsi_4_limit = IntParameter(0, 50, default=buy_params['lambo1_rsi_4_limit'], space='buy', optimize=True)
    lambo1_rsi_14_limit = IntParameter(0, 50, default=buy_params['lambo1_rsi_14_limit'], space='buy', optimize=True)

    # hard stoploss profit
    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    # come from BB_RPB_TSL
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Set Up Bollinger Bands
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        # Clucha

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        # lambo1
        dataframe['ema_14'] = ta.EMA(dataframe['ha_close'], timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe['ha_close'], timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe['ha_close'], timeperiod=14)

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        lambo1 = (
            bool(self.lambo1_enabled.value) &
            (dataframe['ha_close'] < (dataframe['ema_14'] * self.lambo1_ema_14_factor.value)) &
            (dataframe['rsi_4'] < self.lambo1_rsi_4_limit.value) &
            (dataframe['rsi_14'] < self.lambo1_rsi_14_limit.value)
        )
        dataframe.loc[lambo1, 'buy_tag'] += 'lambo1_'
        conditions.append(lambo1)

        clucHA = (
            bool(self.clucha_enabled.value) &
            (dataframe['rocr_1h'].gt(self.rocr_1h.value)) &
            ((
                     (dataframe['lower'].shift().gt(0)) &
                     (dataframe['bbdelta'].gt(dataframe['ha_close'] * self.bbdelta_close.value)) &
                     (dataframe['closedelta'].gt(dataframe['ha_close'] * self.closedelta_close.value)) &
                     (dataframe['tail'].lt(dataframe['bbdelta'] * self.bbdelta_tail.value)) &
                     (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                     (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
             ) |
             (
                     (dataframe['ha_close'] < dataframe['ema_slow']) &
                     (dataframe['ha_close'] < self.close_bblower.value * dataframe['bb_lowerband'])
             ))
        )
        dataframe.loc[clucHA, 'buy_tag'] += 'clucHA_'
        conditions.append(clucHA)

        dataframe.loc[
            reduce(lambda x, y: x | y, conditions),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), 'sell'] = 1
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        trade.sell_reason = sell_reason + "_" + trade.buy_tag

        return True
