# -*- coding: utf-8 -*-
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
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

import math
import pandas_ta as pta

######################################## Warning ########################################
# You won't get a lot of benefits by simply changing to this strategy                   #
# with the HyperOpt values changed.                                                     #
#                                                                                       #
# You should test it closely, trying backtesting and dry running, and we recommend      #
# customizing the terms of sale and purchase as well.                                   #
#                                                                                       #
# You should always be careful in real trading!                                         #
#########################################################################################

# Modified Buy / Sell params - 20210619
# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 16,
    "ewo_high": 5.672,
    "ewo_low": -19.931,
    "low_offset": 0.973,
    "rsi_buy": 59,
    "ewo_high_bb": 4.86,
    "for_ma_length": 22,
    "for_sigma": 1.74,
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 20,
    "high_offset": 1.010,
    "pHSL": -0.178,
    "pPF_1": 0.01,
    "pPF_2": 0.048,
    "pSL_1": 0.009,
    "pSL_2": 0.043,
    "for_ma_length_sell": 65,
    "for_sigma_sell": 1.895,
    "rsi_high": 72,
}

# Volume Weighted Moving Average
def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma


# Modified Elder Ray Index
def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class SMA_BBRSI(IStrategy):
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

    antipump_threshold = DecimalParameter(0, 0.4, default=0.25, space='buy', optimize=True)

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
    startup_candle_count: int = 200

    plot_config = {
        'main_plot': {
            'ma_buy_16': {'color': 'orange'},
            'ma_sell_20': {'color': 'purple'},
        },
            'pump_strength': {
                'pump_strength': {'color': 'yellow'}
            },
    }

    protections = [
        #   {
        #       "method": "StoplossGuard",
        #       "lookback_period_candles": 12,
        #       "trade_limit": 1,
        #       "stop_duration_candles": 6,
        #       "only_per_pair": True
        #   },
        #   {
        #       "method": "StoplossGuard",
        #       "lookback_period_candles": 12,
        #       "trade_limit": 2,
        #       "stop_duration_candles": 6,
        #       "only_per_pair": False
        #   },
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

    ewo_high_bb = DecimalParameter(0, 7.0, default=buy_params['ewo_high_bb'], space='buy', optimize=True)
    for_sigma = DecimalParameter(0, 10.0, default=buy_params['for_sigma'], space='buy', optimize=True)
    for_sigma_sell = DecimalParameter(0, 10.0, default=sell_params['for_sigma_sell'], space='sell', optimize=True)
    rsi_high = IntParameter(60, 100, default=sell_params['rsi_high'], space='sell', optimize=True)
    for_ma_length = IntParameter(5, 80, default=buy_params['for_ma_length'], space='buy', optimize=True)
    for_ma_length_sell = IntParameter(5, 80, default=sell_params['for_ma_length_sell'], space='sell', optimize=True)


    is_optimize_trailing = True
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)

    use_custom_stoploss = True

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

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe




    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # //@version=3
        # study(" RSI + BB (EMA) + Dispersion (2.0)", overlay=false)
        #
        # // Инициализация параметров
        # src = input(title="Source", type=source, defval=close) // Устанавливаем тип цены для расчетов
        src = 'close'
        # for_rsi = input(title="RSI_period", type=integer, defval=14) // Период для RSI
        for_rsi = 14
        # for_ma = input(title="Basis_BB", type=integer, defval=20) // Период для MA внутри BB
        # for_ma = 20
        # for_mult = input(title="Stdev", type=integer, defval=2, minval=1, maxval=5) // Число стандартных отклонений для BB
        for_mult = 2
        # for_sigma = input(title="Dispersion", type=float, defval=0.1, minval=0.01, maxval=1) // Дисперсия вокруг MA
        for_sigma = 0.1
        #
        # // Условия работы скрипта
        # current_rsi = rsi(src, for_rsi) // Текущее положение индикатора RSI
        dataframe['rsi'] = ta.RSI(dataframe[src], for_rsi)
        dataframe['rsi_4'] = ta.RSI(dataframe[src], 4)
        if self.config['runmode'].value == 'hyperopt':
            for for_ma in range(5, 81):
                # basis = ema(current_rsi, for_ma)
                dataframe[f'basis_{for_ma}'] = ta.EMA(dataframe['rsi'], for_ma)
                # dev = for_mult * stdev(current_rsi, for_ma)
                dataframe[f'dev_{for_ma}'] = ta.STDDEV(dataframe['rsi'], for_ma)
                # upper = basis + dev
                #dataframe[f'upper_{for_ma}'] = (dataframe[f'basis_{for_ma}'] + (dataframe[f'dev_{for_ma}'] * for_mult))
                # lower = basis - dev
                #dataframe[f'lower_{for_ma}'] = dataframe[f'basis_{for_ma}'] - (dataframe[f'dev_{for_ma}'] * for_mult)
                # disp_up = basis + ((upper - lower) * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (сверху)
                # dataframe[f'disp_up_{for_ma}'] = dataframe[f'basis_{for_ma}'] + ((dataframe[f'upper_{for_ma}'] - dataframe[f'lower_{for_ma}']) * for_sigma)
                # disp_down = basis - ((upper - lower) * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (снизу)
                # dataframe[f'disp_down_{for_ma}'] = dataframe[f'basis_{for_ma}'] - ((dataframe[f'upper_{for_ma}'] - dataframe[f'lower_{for_ma}']) * for_sigma)
                # color_rsi = current_rsi >= disp_up ? lime : current_rsi <= disp_down ? red : #ffea00 // Текущий цвет RSI, в зависимости от его местоположения внутри BB
        else:
            dataframe[f'basis_{self.for_ma_length.value}'] = ta.EMA(dataframe['rsi'], self.for_ma_length.value)
            dataframe[f'basis_{self.for_ma_length_sell.value}'] = ta.EMA(dataframe['rsi'], self.for_ma_length_sell.value)
            # dev = for_mult * stdev(current_rsi, for_ma)
            dataframe[f'dev_{self.for_ma_length.value}'] = ta.STDDEV(dataframe['rsi'], self.for_ma_length.value)
            dataframe[f'dev_{self.for_ma_length_sell.value}'] = ta.STDDEV(dataframe['rsi'], self.for_ma_length_sell.value)

        #
        # // Дополнительные линии и заливка для областей для RSI
        # h1 = hline(70, color=#d4d4d4, linestyle=dotted, linewidth=1)
        h1 = 70
        # h2 = hline(30, color=#d4d4d4, linestyle=dotted, linewidth=1)
        h2 = 30
        # fill (h1, h2, transp=95)
        #
        # // Алерты и условия срабатывания
        # rsi_Green = crossover(current_rsi, disp_up)
        # rsi_Red = crossunder(current_rsi, disp_down)

        # alertcondition(condition=rsi_Green,
        #      title="RSI cross Above Dispersion Area",
        #      message="The RSI line closing crossed above the Dispersion area.")
        #
        # alertcondition(condition=rsi_Red,
        #      title="RSI cross Under Dispersion Area",
        #      message="The RSI line closing crossed below the Dispersion area")
        #
        # // Результаты и покраска
        # plot(basis, color=black)
        # plot(upper, color=#00fff0, linewidth=2)
        # plot(lower, color=#00fff0, linewidth=2)
        # s1 = plot(disp_up, color=white)
        # s2 = plot(disp_down, color=white)
        # fill(s1, s2, color=white, transp=80)
        # plot(current_rsi, color=color_rsi, linewidth=2)

        #ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_high'] = (dataframe['high'] - (dataframe['atr'] * 3.5)).rolling(2).max()
        dataframe['ema_atr'] = ta.EMA(dataframe['atr'], timeperiod=14)

#        #HLC3
#        dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3

#        #OTF
#        dataframe['hlc3OTF'] = OTF(dataframe, source='close')

#        #ZLEMA BUY
#        dataframe['zlema_1'] = dataframe['hlc3OTF']
#        dataframe['zlema_1_std'] =  dataframe['close']
#        dataframe['ema_data'] = dataframe['hlc3OTF']  + (dataframe['hlc3OTF']  - dataframe['hlc3OTF'].shift(2))
#        dataframe['ema_data_2'] = dataframe['hlc3OTF']  + (dataframe['hlc3OTF']  - dataframe['hlc3OTF'].shift(1))
#        dataframe['zlema_4'] = ta.EMA(dataframe['ema_data'], timeperiod = 4)
#        dataframe['zlema_2'] = ta.EMA(dataframe['ema_data_2'], timeperiod = 2)
#        dataframe['ema_data_4_std'] = dataframe['close']  + (dataframe['close']  - dataframe['close'].shift(2))
#        dataframe['zlema_4_std'] = ta.EMA(dataframe['ema_data_4_std'], timeperiod = 4)

        # Modified Elder Ray Index
        dataframe['moderi_96'] = moderi(dataframe, 96)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        #CCI
        dataframe['cci_slow'] = pta.cci(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=240)
        dataframe['cci_fast'] = pta.cci(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=20)


        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        #pump stregth
        dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dont_buy_conditions = []

        dont_buy_conditions.append(
            (dataframe['pump_strength'] > self.antipump_threshold.value)
        )

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

        conditions.append(
            (
                (dataframe['rsi'] < (dataframe[f'basis_{self.for_ma_length.value}'] - (dataframe[f'dev_{self.for_ma_length.value}'] * self.for_sigma.value)))
                &
                (
                    (
                        (dataframe['EWO'] > self.ewo_high_bb.value)
                        &
                        (dataframe['EWO'] < 10)
                    )
                    |
                    (
                        (dataframe['EWO'] >= 10)
                        &
                        (dataframe['rsi'] < 40)
                    )
                )
                &
                (dataframe['rsi_4'] < 25)
                &
                (dataframe['volume'] > 0)
                &
                (dataframe['cci_fast'] < 100)
#                &
#                (qtpylib.crossed_above(dataframe['hlc3OTF'], dataframe['zlema_4']))
                # &
                # (dataframe["roc_bbwidth_max"] < 70)
            ) & (dataframe['moderi_96'] == True)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                (((
                    (dataframe['rsi'] > self.rsi_high.value) |
                    # upper = basis + dev
                    # lower = basis - dev
                    # disp_down = basis - ((upper - lower) * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (снизу)
                    # disp_down = basis - ((2* dev * for_sigma) // Минимально-допустимый порог в области мувинга, который должен преодолеть RSI (снизу)
                    (dataframe['rsi'] > dataframe[f'basis_{self.for_ma_length_sell.value}'] + ((dataframe[f'dev_{self.for_ma_length_sell.value}'] * self.for_sigma_sell.value)))
                ) & (dataframe['moderi_96'] == True)) |
                (
                    (qtpylib.crossed_below(dataframe['close'], dataframe['atr_high']))
                ))&
                (dataframe['volume'] > 0)

            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
