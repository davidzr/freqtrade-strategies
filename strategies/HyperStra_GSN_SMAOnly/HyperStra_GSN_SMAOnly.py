from logging import FATAL
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt

from talib import abstract

# ###############################################################################
# ###############################################################################
# @Farhad#0318
# Idea is from GodStraNew_SMAOnly
# ###############################################################################
# ###############################################################################

# MA Indicator ( EMA, SMA, MA, ... )
MA_Indicator = abstract.SMA

class HyperStra_GSN_SMAOnly(IStrategy):
    INTERFACE_VERSION = 2

    # ##################################################################
    # Hyperopt Params Paste Here

    # Buy hyperspace params:
    buy_params = {
        "buy_1_indicator": 5,
        "buy_1_indicator_sec": 110,
        "buy_1_operator": "normalized_devided_smaller_n",
        "buy_1_real_number": 0.3,

        "buy_2_indicator": 5,
        "buy_2_indicator_sec": 6,
        "buy_2_operator": "normalized_smaller_n",
        "buy_2_real_number": 0.5,

        "buy_3_indicator": 55,
        "buy_3_indicator_sec": 100,
        "buy_3_operator": "normalized_devided_smaller_n",
        "buy_3_real_number": 0.9,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_1_indicator": 50,
        "sell_1_indicator_sec": 15,
        "sell_1_operator": "equal",
        "sell_1_real_number": 0.9,

        "sell_2_indicator": 50,
        "sell_2_indicator_sec": 110,
        "sell_2_operator": "cross_above",
        "sell_2_real_number": 0.5,

        "sell_3_indicator": 110,
        "sell_3_indicator_sec": 5,
        "sell_3_operator": "below",
        "sell_3_real_number": 0.8,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.288,
        "81": 0.101,
        "170": 0.049,
        "491": 0
    }

    # Stoploss:
    stoploss = -0.05

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    # ##################################################################
    # ##################################################################

    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 2,
                "trade_limit": 1,
                "stop_duration_candles": 12,
                "only_per_pair": False
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2
            }
        ]

    # ##################################################################
    # ##################################################################

    # Sell signal
    use_custom_stoploss = False
    use_sell_signal = True
    timeframe = '5m'
    ignore_roi_if_buy_signal = False
    process_only_new_candles = False
    startup_candle_count = 440

    sell_profit_only = False
    sell_profit_offset = 0.01

    # ##################################################################
    # ##################################################################

    # #################################
    # Optimiztions HyperSMA BUY
    optimize_hypersma_buy_1_1_sma = True
    optimize_hypersma_buy_1_2_sma = True
    optimize_hypersma_buy_1_3_sma = True

    # #################################
    # Optimiztions HyperSMA Sell
    optimize_hypersma_sell_1_1_sma = True
    optimize_hypersma_sell_1_2_sma = True
    optimize_hypersma_sell_1_3_sma = True

    # ##################################################################
    # ##################################################################

    sma_timeperiods = [5, 6, 15, 50, 55, 100, 110]
    sma_operators = [
        'equal',
        'above',
        'below',
        'cross_above',
        'cross_below',
        'divide_greater',
        'divide_smaller',
        'normalized_equal_n',
        'normalized_smaller_n',
        'normalized_bigger_n',
        'normalized_devided_equal_n',
        'normalized_devided_smaller_n',
        'normalized_devided_bigger_n'
    ]

    # ##################################################################
    # ##################################################################
    # HyperSMA

    # normalizer_lenght = IntParameter(low=1, high=400, default=20, space='buy', optimize=True)

    # BUY
    buy_1_indicator = CategoricalParameter(categories=sma_timeperiods, default=buy_params['buy_1_indicator'], space='buy', optimize=optimize_hypersma_buy_1_1_sma)
    buy_1_indicator_sec = CategoricalParameter(categories=sma_timeperiods, default=buy_params['buy_1_indicator_sec'], space='buy', optimize=optimize_hypersma_buy_1_1_sma)
    buy_1_real_number = DecimalParameter(low=0, high=0.99, default=buy_params['buy_1_real_number'], decimals=2, space='buy', optimize=optimize_hypersma_buy_1_1_sma)
    buy_1_operator = CategoricalParameter(categories=sma_operators, default=buy_params['buy_1_operator'], space='buy', optimize=optimize_hypersma_buy_1_1_sma)

    buy_2_indicator = CategoricalParameter(categories=sma_timeperiods, default=buy_params['buy_2_indicator'], space='buy', optimize=optimize_hypersma_buy_1_2_sma)
    buy_2_indicator_sec = CategoricalParameter(categories=sma_timeperiods, default=buy_params['buy_2_indicator_sec'], space='buy', optimize=optimize_hypersma_buy_1_2_sma)
    buy_2_real_number = DecimalParameter(low=0, high=0.99, default=buy_params['buy_2_real_number'], decimals=2, space='buy', optimize=optimize_hypersma_buy_1_2_sma)
    buy_2_operator = CategoricalParameter(categories=sma_operators, default=buy_params['buy_2_operator'], space='buy', optimize=optimize_hypersma_buy_1_2_sma)

    buy_3_indicator = CategoricalParameter(categories=sma_timeperiods, default=buy_params['buy_3_indicator'], space='buy', optimize=optimize_hypersma_buy_1_3_sma)
    buy_3_indicator_sec = CategoricalParameter(categories=sma_timeperiods, default=buy_params['buy_3_indicator_sec'], space='buy', optimize=optimize_hypersma_buy_1_3_sma)
    buy_3_real_number = DecimalParameter(low=0, high=0.99, default=buy_params['buy_3_real_number'], decimals=2, space='buy', optimize=optimize_hypersma_buy_1_3_sma)
    buy_3_operator = CategoricalParameter(categories=sma_operators, default=buy_params['buy_3_operator'], space='buy', optimize=optimize_hypersma_buy_1_3_sma)

    # SELL
    sell_1_indicator = CategoricalParameter(categories=sma_timeperiods, default=sell_params['sell_1_indicator'], space='sell', optimize=optimize_hypersma_sell_1_1_sma)
    sell_1_indicator_sec = CategoricalParameter(categories=sma_timeperiods, default=sell_params['sell_1_indicator_sec'], space='sell', optimize=optimize_hypersma_sell_1_1_sma)
    sell_1_real_number = DecimalParameter(low=0, high=0.99, default=sell_params['sell_1_real_number'], decimals=2, space='sell', optimize=optimize_hypersma_sell_1_1_sma)
    sell_1_operator = CategoricalParameter(categories=sma_operators, default=sell_params['sell_1_operator'], space='sell', optimize=optimize_hypersma_sell_1_1_sma)

    sell_2_indicator = CategoricalParameter(categories=sma_timeperiods, default=sell_params['sell_2_indicator'], space='sell', optimize=optimize_hypersma_sell_1_2_sma)
    sell_2_indicator_sec = CategoricalParameter(categories=sma_timeperiods, default=sell_params['sell_2_indicator_sec'], space='sell', optimize=optimize_hypersma_sell_1_2_sma)
    sell_2_real_number = DecimalParameter(low=0, high=0.99, default=sell_params['sell_2_real_number'], decimals=2, space='sell', optimize=optimize_hypersma_sell_1_2_sma)
    sell_2_operator = CategoricalParameter(categories=sma_operators, default=sell_params['sell_2_operator'], space='sell', optimize=optimize_hypersma_sell_1_2_sma)

    sell_3_indicator = CategoricalParameter(categories=sma_timeperiods, default=sell_params['sell_3_indicator'], space='sell', optimize=optimize_hypersma_sell_1_3_sma)
    sell_3_indicator_sec = CategoricalParameter(categories=sma_timeperiods, default=sell_params['sell_3_indicator_sec'], space='sell', optimize=optimize_hypersma_sell_1_3_sma)
    sell_3_real_number = DecimalParameter(low=0, high=0.99, default=sell_params['sell_3_real_number'], decimals=2, space='sell', optimize=optimize_hypersma_sell_1_3_sma)
    sell_3_operator = CategoricalParameter(categories=sma_operators, default=sell_params['sell_3_operator'], space='sell', optimize=optimize_hypersma_sell_1_3_sma)

    # ##################################################################
    # HyperSMA
    # ##################################################################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ###############################
        # Multi SMA
        for m_timeperiod in self.sma_timeperiods:
            dataframe[f'ma_{m_timeperiod}'] = MA_Indicator(dataframe, timeperiod=m_timeperiod)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                    (dataframe['volume'] > 0) &
                    (
                            (
                                self.condition_maker(dataframe=dataframe,
                                                     indicator=self.buy_1_indicator.value,
                                                     indicator_sec=self.buy_1_indicator_sec.value,
                                                     real_number=self.buy_1_real_number.value,
                                                     operator=self.buy_1_operator.value, option='buy')
                            )
                            &
                            (
                                self.condition_maker(dataframe=dataframe,
                                                     indicator=self.buy_2_indicator.value,
                                                     indicator_sec=self.buy_2_indicator_sec.value,
                                                     real_number=self.buy_2_real_number.value,
                                                     operator=self.buy_2_operator.value, option='buy')
                            )
                            &
                            (
                                self.condition_maker(dataframe=dataframe,
                                                     indicator=self.buy_3_indicator.value,
                                                     indicator_sec=self.buy_3_indicator_sec.value,
                                                     real_number=self.buy_3_real_number.value,
                                                     operator=self.buy_3_operator.value, option='buy')
                            )
                    )
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                    (dataframe['volume'] > 0) &
                    (
                            (
                                self.condition_maker(dataframe=dataframe,
                                                     indicator=self.sell_1_indicator.value, indicator_sec=self.sell_1_indicator_sec.value,
                                                     real_number=self.sell_1_real_number.value,
                                                     operator=self.sell_1_operator.value, option='sell')
                            )
                            &
                            (
                                self.condition_maker(dataframe=dataframe,
                                                     indicator=self.sell_2_indicator.value, indicator_sec=self.sell_2_indicator_sec.value,
                                                     real_number=self.sell_2_real_number.value,
                                                     operator=self.sell_2_operator.value, option='sell')
                            )
                            &
                            (
                                self.condition_maker(dataframe=dataframe,
                                                     indicator=self.sell_3_indicator.value, indicator_sec=self.sell_3_indicator_sec.value,
                                                     real_number=self.sell_3_real_number.value,
                                                     operator=self.sell_3_operator.value, option='sell')
                            )
                    )
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'sell'] = 1
        return dataframe

    def condition_maker(self, dataframe: DataFrame, indicator: int, indicator_sec: int, real_number: float, operator: str, option: str):
        indicator_1 = f'ma_{indicator}'
        indicator_2 = f'ma_{indicator_sec}'

        if operator == 'equal':
            return dataframe[indicator_1] == dataframe[indicator_2]

        if operator == 'above':
            return dataframe[indicator_1] >= dataframe[indicator_2]

        if operator == 'below':
            return dataframe[indicator_1] <= dataframe[indicator_2]

        if operator == 'cross_above':
            return qtpylib.crossed_above(dataframe[indicator_1], dataframe[indicator_2])

        if operator == 'cross_below':
            return qtpylib.crossed_below(dataframe[indicator_1], dataframe[indicator_2])

        if operator == 'divide_greater':
            return dataframe[indicator_1].div(dataframe[indicator_2]) <= real_number

        if operator == 'divide_smaller':
            return dataframe[indicator_1].div(dataframe[indicator_2]) >= real_number

        if operator == 'normalized_equal_n':
            return Normalizer(dataframe[indicator_1]) == real_number

        if operator == 'normalized_smaller_n':
            return Normalizer(dataframe[indicator_1]) < real_number

        if operator == 'normalized_bigger_n':
            return Normalizer(dataframe[indicator_1]) > real_number

        if operator == 'normalized_devided_equal_n':
            return (Normalizer(dataframe[indicator_1]).div(Normalizer(dataframe[indicator_2]))) == real_number

        if operator == 'normalized_devided_smaller_n':
            return (Normalizer(dataframe[indicator_1]).div(Normalizer(dataframe[indicator_2]))) < real_number

        if operator == 'normalized_devided_bigger_n':
            return (Normalizer(dataframe[indicator_1]).div(Normalizer(dataframe[indicator_2]))) > real_number


# ##################################################################
# Methods
# ##################################################################

def Normalizer(df: DataFrame) -> DataFrame:
    df = (df - df.min()) / (df.max() - df.min())
    return df
