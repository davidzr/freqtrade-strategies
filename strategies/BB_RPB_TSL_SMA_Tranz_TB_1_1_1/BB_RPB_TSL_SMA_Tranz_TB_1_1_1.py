# --- Do not remove these libs ---
import pandas_ta as pta
import copy
import logging
import pathlib
import rapidjson
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, timeframe_to_minutes
from freqtrade.exchange import timeframe_to_prev_date
from pandas import DataFrame, Series, concat
from functools import reduce
import math
from random import shuffle
from typing import Dict
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from technical.util import resample_to_interval, resampled_merge
from technical.indicators import RMI, zema, VIDYA, ichimoku
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
import time

log = logging.getLogger(__name__)

# --------------------------------
def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

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
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc

def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from âˆ’100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

class BB_RPB_TSL_SMA_Tranz_TB_1_1_1(IStrategy):
    '''
        BB_RPB_TSL
        @author jilv220
        Simple bollinger brand strategy inspired by this blog  ( https://hacks-for-life.blogspot.com/2020/12/freqtrade-notes.html )
        RPB, which stands for Real Pull Back, taken from ( https://github.com/GeorgeMurAlkh/freqtrade-stuff/blob/main/user_data/strategies/TheRealPullbackV2.py )
        The trailing custom stoploss taken from BigZ04_TSL from Perkmeister ( modded by ilya )
        I modified it to better suit my taste and added Hyperopt for this strategy.
    '''

    # (1) sell rework

    ##########################################################################

    # Hyperopt result area

    # buy space
    buy_params = {
        "max_slip": 0.983,
        ##
        "buy_bb_width_1h": 0.954,
        "buy_roc_1h": 86,
        ##
        "buy_threshold": 0.003,
        "buy_bb_factor": 0.999,
        #
        "buy_bb_delta": 0.025,
        "buy_bb_width": 0.095,
        ##
        "buy_cci": -116,
        "buy_cci_length": 25,
        "buy_rmi": 49,
        "buy_rmi_length": 17,
        "buy_srsi_fk": 32,
        ##
        "buy_closedelta": 17.922,
        "buy_ema_diff": 0.026,
        ##
        "buy_ema_high": 0.968,
        "buy_ema_low": 0.935,
        "buy_ewo": -5.001,
        "buy_rsi": 23,
        "buy_rsi_fast": 44,
        ##
        "buy_closedelta_local_dip": 12.044,
        "buy_ema_diff_local_dip": 0.024,
        "buy_ema_high_local_dip": 1.014,
        "buy_rsi_local_dip": 21,
        ##
        "buy_r_deadfish_bb_factor": 1.014,
        "buy_r_deadfish_bb_width": 0.299,
        "buy_r_deadfish_ema": 1.054,
        "buy_r_deadfish_volume_factor": 1.59,
        "buy_r_deadfish_cti": -0.115,
        "buy_r_deadfish_r14": -44.34,
        ##
        "buy_clucha_bbdelta_close": 0.04796,
        "buy_clucha_bbdelta_tail": 0.93112,
        "buy_clucha_close_bblower": 0.01645,
        "buy_clucha_closedelta_close": 0.00931,
        "buy_clucha_rocr_1h": 0.41663,
        ##
        "buy_ema_high_2": 1.04116,
        "buy_ema_low_2": 0.97463,
        "buy_ewo_high_2": 5.249,
        "buy_rsi_ewo_2": 35,
        "buy_rsi_fast_ewo_2": 45,
        ##
        "buy_adx": 13,
        "buy_cofi_r14": -85.016,
        "buy_cofi_cti": -0.892,
        "buy_ema_cofi": 1.147,
        "buy_ewo_high": 8.594,
        "buy_fastd": 28,
        "buy_fastk": 39,
        ##
        "buy_gumbo_ema": 1.121,
        "buy_gumbo_ewo_low": -9.442,
        "buy_gumbo_cti": -0.374,
        "buy_gumbo_r14": -51.971,
        ##
        "buy_sqzmom_ema": 0.981,
        "buy_sqzmom_ewo": -3.966,
        "buy_sqzmom_r14": -45.068,
        ##
        "buy_nfix_49_cti": -0.105,
        "buy_nfix_49_r14": -81.827,
        ##DIAMOND
        "buy_fast": 31,
        "buy_push": 0.72,
        "buy_shift": -7,
        "buy_slow": 2,
    }
    #############################################################
    sell_params = {
        ##
        "sell_cmf": -0.046,
        "sell_ema": 0.988,
        "sell_ema_close_delta": 0.022,
        ##
        "sell_deadfish_profit": -0.063,
        "sell_deadfish_bb_factor": 0.954,
        "sell_deadfish_bb_width": 0.043,
        "sell_deadfish_volume_factor": 2.37,
        ##
        "sell_cti_r_cti": 0.844,
        "sell_cti_r_r": -19.99,
        #DIAMOND
        "sell_fast": 17,
        "sell_push": 1.493,
        "sell_shift": -7,
        "sell_slow": 28,
        #############
        # Enable/Disable conditions
        "sell_condition_1_enable": True,
        "sell_condition_2_enable": True,
        "sell_condition_3_enable": True,
        "sell_condition_4_enable": True,
        "sell_condition_5_enable": True,
        "sell_condition_6_enable": True,
        "sell_condition_7_enable": True,
        "sell_condition_8_enable": True,
        #############
    }

    sell_condition_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_5_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_6_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_7_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_8_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)


    minimal_roi = {
        "0": 0.205,
        "81": 0.038,
        "292": 0.005,
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'
    inf_15m = '15m'
    info_timefame_1d = 'none'
    info_timeframe_1h = '1h'
    info_timeframe_15m = '15m'
    res_timeframe = 'none'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Disabled
    stoploss = -0.99

    # Custom stoploss
    use_custom_stoploss = True
    use_sell_signal = True

    startup_candle_count: int = 400
    ############################################################################

    ## Buy params

    is_optimize_dip = False
    buy_rmi = IntParameter(30, 50, default=35, optimize= is_optimize_dip)
    buy_cci = IntParameter(-135, -90, default=-133, optimize= is_optimize_dip)
    buy_srsi_fk = IntParameter(30, 50, default=25, optimize= is_optimize_dip)
    buy_cci_length = IntParameter(25, 45, default=25, optimize = is_optimize_dip)
    buy_rmi_length = IntParameter(8, 20, default=8, optimize = is_optimize_dip)

    is_optimize_break = False
    buy_bb_width = DecimalParameter(0.065, 0.135, default=0.095, optimize = is_optimize_break)
    buy_bb_delta = DecimalParameter(0.018, 0.035, default=0.025, optimize = is_optimize_break)

    is_optimize_local_uptrend = False
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_uptrend)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize = False)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_uptrend)

    is_optimize_local_dip = False
    buy_ema_diff_local_dip = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_dip)
    buy_ema_high_local_dip = DecimalParameter(0.90, 1.2, default=0.942 , optimize = is_optimize_local_dip)
    buy_closedelta_local_dip = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_dip)
    buy_rsi_local_dip = IntParameter(15, 45, default=28, optimize = is_optimize_local_dip)
    buy_crsi_local_dip = IntParameter(10, 18, default=10, optimize = False)

    is_optimize_ewo = False
    buy_rsi_fast = IntParameter(35, 50, default=45, optimize = is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=35, optimize = is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.585, optimize = is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942 , optimize = is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.084 , optimize = is_optimize_ewo)

    is_optimize_r_deadfish = False
    buy_r_deadfish_ema = DecimalParameter(0.90, 1.2, default=1.087 , optimize = is_optimize_r_deadfish)
    buy_r_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , optimize = is_optimize_r_deadfish)
    buy_r_deadfish_bb_factor = DecimalParameter(0.90, 1.2, default=1.0 , optimize = is_optimize_r_deadfish)
    buy_r_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_r_deadfish)

    is_optimize_r_deadfish_protection = False
    buy_r_deadfish_cti = DecimalParameter(-0.6, -0.0, default=-0.5 , optimize = is_optimize_r_deadfish_protection)
    buy_r_deadfish_r14 = DecimalParameter(-60, -44, default=-60 , optimize = is_optimize_r_deadfish_protection)

    is_optimize_clucha = False
    buy_clucha_bbdelta_close = DecimalParameter(0.01,0.05, default=0.02206, optimize = is_optimize_clucha)
    buy_clucha_bbdelta_tail = DecimalParameter(0.7, 1.2, default=1.02515, optimize = is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.001, 0.05, default=0.04401, optimize = is_optimize_clucha)
    buy_clucha_rocr_1h = DecimalParameter(0.1, 1.0, default=0.47782, optimize = is_optimize_clucha)

    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.94, 1.2, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(0, 40, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(0, 40, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(0, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    is_optimize_cofi_protection = False
    buy_cofi_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_cofi_protection)
    buy_cofi_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_cofi_protection)

    is_optimize_gumbo = False
    buy_gumbo_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_gumbo)
    buy_gumbo_ewo_low = DecimalParameter(-12.0, 5, default=-5.585, optimize = is_optimize_gumbo)

    is_optimize_gumbo_protection = False
    buy_gumbo_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_gumbo_protection)
    buy_gumbo_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_gumbo_protection)

    is_optimize_sqzmom_protection = False
    buy_sqzmom_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_sqzmom_protection)
    buy_sqzmom_ewo = DecimalParameter(-12 , 12, default= 0 , optimize = is_optimize_sqzmom_protection)
    buy_sqzmom_r14 = DecimalParameter(-100, -22, default=-50 , optimize = is_optimize_sqzmom_protection)

    is_optimize_nfix_39 = True
    buy_nfix_39_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_nfix_39)

    is_optimize_nfix_49_protection = False
    buy_nfix_49_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_nfix_49_protection)
    buy_nfix_49_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_nfix_49_protection)

    is_optimize_btc_safe = False
    buy_btc_safe = IntParameter(-300, 50, default=-200, optimize = is_optimize_btc_safe)
    buy_btc_safe_1d = DecimalParameter(-0.075, -0.025, default=-0.05, optimize = is_optimize_btc_safe)
    buy_threshold = DecimalParameter(0.003, 0.012, default=0.008, optimize = is_optimize_btc_safe)

    is_optimize_check = False
    buy_roc_1h = IntParameter(-25, 200, default=10, optimize = is_optimize_check)
    buy_bb_width_1h = DecimalParameter(0.3, 2.0, default=0.3, optimize = is_optimize_check)

    #BB MODDED
    is_optimize_ctt15_protection = False
    buy_ema_open_mult_15 = DecimalParameter(0.01, 0.03, default=0.024, optimize = is_optimize_ctt15_protection)
    buy_ma_offset_15 = DecimalParameter(0.93, 0.99, default=0.958, optimize = is_optimize_ctt15_protection)
    buy_rsi_15 = DecimalParameter(20.0, 36.0, default=28.0, optimize = is_optimize_ctt15_protection)
    buy_ema_rel_15 = DecimalParameter(0.97, 0.999, default=0.974, optimize = is_optimize_ctt15_protection)

    is_optimize_ctt25_protection = False
    buy_25_ma_offset = DecimalParameter(0.90, 0.99, default=0.922, optimize = is_optimize_ctt25_protection)
    buy_25_rsi_4 = DecimalParameter(26.0, 40.0, default=38.0, optimize = is_optimize_ctt25_protection)
    buy_25_cti = DecimalParameter(-0.99, -0.4, default=-0.76, optimize = is_optimize_ctt25_protection)

    #DIAMOND
    is_diamond_protection = False
    buy_push = DecimalParameter(0, 2, decimals=3, default=1, optimize = is_diamond_protection)
    buy_shift = IntParameter(-10, 0, default=-6, optimize = is_diamond_protection)
    buy_fast = IntParameter(2, 50, default=9, optimize = is_diamond_protection)
    buy_slow = IntParameter(2, 50, default=18, optimize = is_diamond_protection)

    #NFI 7 SMA
    buy_dip_threshold_10_1 = DecimalParameter(0.001, 0.05, default=0.015, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_10_2 = DecimalParameter(0.01, 0.2, default=0.1, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_10_3 = DecimalParameter(0.1, 0.3, default=0.24, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_10_4 = DecimalParameter(0.3, 0.5, default=0.42, space='buy', decimals=3, optimize=False, load=True)
    # Strict dips - level 20
    buy_dip_threshold_20_1 = DecimalParameter(0.001, 0.05, default=0.016, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_20_2 = DecimalParameter(0.01, 0.2, default=0.11, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_20_3 = DecimalParameter(0.1, 0.4, default=0.26, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_20_4 = DecimalParameter(0.36, 0.56, default=0.44, space='buy', decimals=3, optimize=False, load=True)
    # Strict dips - level 30
    buy_dip_threshold_30_1 = DecimalParameter(0.001, 0.05, default=0.018, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_30_2 = DecimalParameter(0.01, 0.2, default=0.12, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_30_3 = DecimalParameter(0.1, 0.4, default=0.28, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_30_4 = DecimalParameter(0.36, 0.56, default=0.46, space='buy', decimals=3, optimize=False, load=True)
    # Strict dips - level 40
    buy_dip_threshold_40_1 = DecimalParameter(0.001, 0.05, default=0.019, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_40_2 = DecimalParameter(0.01, 0.2, default=0.13, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_40_3 = DecimalParameter(0.1, 0.4, default=0.3, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_40_4 = DecimalParameter(0.36, 0.56, default=0.48, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 50
    buy_dip_threshold_50_1 = DecimalParameter(0.001, 0.05, default=0.02, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_50_2 = DecimalParameter(0.01, 0.2, default=0.14, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_50_3 = DecimalParameter(0.05, 0.4, default=0.32, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_50_4 = DecimalParameter(0.2, 0.5, default=0.5, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 60
    buy_dip_threshold_60_1 = DecimalParameter(0.001, 0.05, default=0.022, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_60_2 = DecimalParameter(0.1, 0.22, default=0.18, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_60_3 = DecimalParameter(0.2, 0.4, default=0.34, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_60_4 = DecimalParameter(0.4, 0.6, default=0.56, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 70
    buy_dip_threshold_70_1 = DecimalParameter(0.001, 0.05, default=0.023, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_70_2 = DecimalParameter(0.16, 0.28, default=0.2, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_70_3 = DecimalParameter(0.2, 0.4, default=0.36, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_70_4 = DecimalParameter(0.5, 0.7, default=0.6, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 80
    buy_dip_threshold_80_1 = DecimalParameter(0.001, 0.05, default=0.024, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_80_2 = DecimalParameter(0.16, 0.28, default=0.22, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_80_3 = DecimalParameter(0.2, 0.4, default=0.38, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_80_4 = DecimalParameter(0.5, 0.7, default=0.66, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 70
    buy_dip_threshold_90_1 = DecimalParameter(0.001, 0.05, default=0.025, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_90_2 = DecimalParameter(0.16, 0.28, default=0.23, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_90_3 = DecimalParameter(0.3, 0.5, default=0.4, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_90_4 = DecimalParameter(0.6, 0.8, default=0.7, space='buy', decimals=3, optimize=False, load=True)
    # Loose dips - level 100
    buy_dip_threshold_100_1 = DecimalParameter(0.001, 0.05, default=0.026, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_100_2 = DecimalParameter(0.16, 0.3, default=0.24, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_100_3 = DecimalParameter(0.3, 0.5, default=0.42, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_100_4 = DecimalParameter(0.6, 1.0, default=0.8, space='buy', decimals=3, optimize=False, load=True)
    # Loose dips - level 110
    buy_dip_threshold_110_1 = DecimalParameter(0.001, 0.05, default=0.027, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_110_2 = DecimalParameter(0.16, 0.3, default=0.26, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_110_3 = DecimalParameter(0.3, 0.5, default=0.44, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_110_4 = DecimalParameter(0.6, 1.0, default=0.84, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 10
    buy_pump_pull_threshold_10_24 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_10_24 = DecimalParameter(0.4, 1.0, default=0.42, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 10
    buy_pump_pull_threshold_10_36 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_10_36 = DecimalParameter(0.4, 1.0, default=0.58, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 10
    buy_pump_pull_threshold_10_48 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_10_48 = DecimalParameter(0.4, 1.0, default=0.8, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 20
    buy_pump_pull_threshold_20_24 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_20_24 = DecimalParameter(0.4, 1.0, default=0.46, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 20
    buy_pump_pull_threshold_20_36 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_20_36 = DecimalParameter(0.4, 1.0, default=0.6, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 20
    buy_pump_pull_threshold_20_48 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_20_48 = DecimalParameter(0.4, 1.0, default=0.81, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 30
    buy_pump_pull_threshold_30_24 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_30_24 = DecimalParameter(0.4, 1.0, default=0.5, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 30
    buy_pump_pull_threshold_30_36 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_30_36 = DecimalParameter(0.4, 1.0, default=0.62, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 30
    buy_pump_pull_threshold_30_48 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_30_48 = DecimalParameter(0.4, 1.0, default=0.82, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 40
    buy_pump_pull_threshold_40_24 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_40_24 = DecimalParameter(0.4, 1.0, default=0.54, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 40
    buy_pump_pull_threshold_40_36 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_40_36 = DecimalParameter(0.4, 1.0, default=0.63, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 40
    buy_pump_pull_threshold_40_48 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_40_48 = DecimalParameter(0.4, 1.0, default=0.84, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 50
    buy_pump_pull_threshold_50_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_50_24 = DecimalParameter(0.4, 1.0, default=0.6, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 50
    buy_pump_pull_threshold_50_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_50_36 = DecimalParameter(0.4, 1.0, default=0.64, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 50
    buy_pump_pull_threshold_50_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_50_48 = DecimalParameter(0.4, 1.0, default=0.85, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 60
    buy_pump_pull_threshold_60_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_60_24 = DecimalParameter(0.4, 1.0, default=0.62, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 60
    buy_pump_pull_threshold_60_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_60_36 = DecimalParameter(0.4, 1.0, default=0.66, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 60
    buy_pump_pull_threshold_60_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_60_48 = DecimalParameter(0.4, 1.0, default=0.9, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 70
    buy_pump_pull_threshold_70_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_70_24 = DecimalParameter(0.4, 1.0, default=0.63, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 70
    buy_pump_pull_threshold_70_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_70_36 = DecimalParameter(0.4, 1.0, default=0.67, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 70
    buy_pump_pull_threshold_70_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_70_48 = DecimalParameter(0.4, 1.0, default=0.95, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 80
    buy_pump_pull_threshold_80_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_80_24 = DecimalParameter(0.4, 1.0, default=0.64, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 80
    buy_pump_pull_threshold_80_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_80_36 = DecimalParameter(0.4, 1.0, default=0.68, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 80
    buy_pump_pull_threshold_80_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_80_48 = DecimalParameter(0.8, 1.1, default=1.0, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 90
    buy_pump_pull_threshold_90_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_90_24 = DecimalParameter(0.4, 1.0, default=0.65, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 90
    buy_pump_pull_threshold_90_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_90_36 = DecimalParameter(0.4, 1.0, default=0.69, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 90
    buy_pump_pull_threshold_90_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_90_48 = DecimalParameter(0.8, 1.2, default=1.1, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 100
    buy_pump_pull_threshold_100_24 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_100_24 = DecimalParameter(0.4, 1.0, default=0.66, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 100
    buy_pump_pull_threshold_100_36 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_100_36 = DecimalParameter(0.4, 1.0, default=0.7, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 100
    buy_pump_pull_threshold_100_48 = DecimalParameter(1.3, 2.0, default=1.4, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_100_48 = DecimalParameter(0.4, 1.8, default=1.6, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 110
    buy_pump_pull_threshold_110_24 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_110_24 = DecimalParameter(0.4, 1.0, default=0.7, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 110
    buy_pump_pull_threshold_110_36 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_110_36 = DecimalParameter(0.4, 1.0, default=0.74, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 110
    buy_pump_pull_threshold_110_48 = DecimalParameter(1.3, 2.0, default=1.4, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_110_48 = DecimalParameter(1.4, 2.0, default=1.8, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 120
    buy_pump_pull_threshold_120_24 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_120_24 = DecimalParameter(0.4, 1.0, default=0.78, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 120
    buy_pump_pull_threshold_120_36 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_120_36 = DecimalParameter(0.4, 1.0, default=0.78, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 120
    buy_pump_pull_threshold_120_48 = DecimalParameter(1.3, 2.0, default=1.4, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_120_48 = DecimalParameter(1.4, 2.8, default=2.0, space='buy', decimals=3, optimize=False, load=True)

    buy_min_inc_1 = DecimalParameter(0.01, 0.05, default=0.022, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_1 = DecimalParameter(25.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_1 = DecimalParameter(70.0, 90.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1 = DecimalParameter(20.0, 40.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_1 = DecimalParameter(20.0, 40.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_1h_min_2 = DecimalParameter(30.0, 40.0, default=32.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_2 = DecimalParameter(70.0, 95.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_diff_2 = DecimalParameter(30.0, 50.0, default=39.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_2 = DecimalParameter(30.0, 56.0, default=49.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_2 = DecimalParameter(0.97, 0.999, default=0.983, space='buy', decimals=3, optimize=False, load=True)

    buy_bb40_bbdelta_close_3 = DecimalParameter(0.005, 0.06, default=0.059, space='buy', optimize=False, load=True)
    buy_bb40_closedelta_close_3 = DecimalParameter(0.01, 0.03, default=0.023, space='buy', optimize=False, load=True)
    buy_bb40_tail_bbdelta_3 = DecimalParameter(0.15, 0.45, default=0.418, space='buy', optimize=False, load=True)
    buy_ema_rel_3 = DecimalParameter(0.97, 0.999, default=0.986, space='buy', decimals=3, optimize=False, load=True)

    buy_bb20_close_bblowerband_4 = DecimalParameter(0.96, 0.99, default=0.98, space='buy', optimize=False, load=True)
    buy_bb20_volume_4 = DecimalParameter(1.0, 20.0, default=10.0, space='buy', decimals=2, optimize=False, load=True)

    buy_ema_open_mult_5 = DecimalParameter(0.016, 0.03, default=0.018, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_5 = DecimalParameter(0.98, 1.0, default=0.996, space='buy', decimals=3, optimize=False, load=True)
    buy_ema_rel_5 = DecimalParameter(0.97, 0.999, default=0.944, space='buy', decimals=3, optimize=False, load=True)

    buy_ema_open_mult_6 = DecimalParameter(0.02, 0.03, default=0.021, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_6 = DecimalParameter(0.98, 0.999, default=0.984, space='buy', decimals=3, optimize=False, load=True)

    buy_ema_open_mult_7 = DecimalParameter(0.02, 0.04, default=0.03, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_7 = DecimalParameter(24.0, 50.0, default=37.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_8 = DecimalParameter(1.0, 6.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_8 = DecimalParameter(16.0, 30.0, default=29.0, space='buy', decimals=1, optimize=False, load=True)
    buy_tail_diff_8 = DecimalParameter(3.0, 10.0, default=3.5, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_9 = DecimalParameter(0.91, 0.94, default=0.922, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_9 = DecimalParameter(0.96, 0.98, default=0.942, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_9 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_9 = DecimalParameter(70.0, 90.0, default=88.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_9 = DecimalParameter(36.0, 56.0, default=50.0, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_10 = DecimalParameter(0.93, 0.97, default=0.948, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_10 = DecimalParameter(0.97, 0.99, default=0.985, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_10 = DecimalParameter(20.0, 40.0, default=37.0, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_11 = DecimalParameter(0.93, 0.99, default=0.934, space='buy', decimals=3, optimize=False, load=True)
    buy_min_inc_11 = DecimalParameter(0.005, 0.05, default=0.01, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_11 = DecimalParameter(40.0, 60.0, default=55.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_11 = DecimalParameter(70.0, 90.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_11 = DecimalParameter(34.0, 50.0, default=48.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_11 = DecimalParameter(30.0, 46.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_12 = DecimalParameter(0.93, 0.97, default=0.922, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_12 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ewo_12 = DecimalParameter(1.0, 6.0, default=1.8, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_13 = DecimalParameter(0.93, 0.98, default=0.99, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_13 = DecimalParameter(-14.0, -7.0, default=-11.4, space='buy', decimals=1, optimize=False, load=True)

    buy_ema_open_mult_14 = DecimalParameter(0.01, 0.03, default=0.014, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_14 = DecimalParameter(0.98, 1.0, default=0.988, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_14 = DecimalParameter(0.93, 0.99, default=0.98, space='buy', decimals=3, optimize=False, load=True)

    buy_ema_open_mult_15 = DecimalParameter(0.01, 0.03, default=0.018, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_15 = DecimalParameter(0.93, 0.99, default=0.954, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_15 = DecimalParameter(20.0, 36.0, default=28.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_rel_15 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=False, load=True)

    buy_ma_offset_16 = DecimalParameter(0.93, 0.97, default=0.952, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_16 = DecimalParameter(26.0, 50.0, default=31.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ewo_16 = DecimalParameter(2.0, 6.0, default=2.8, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_17 = DecimalParameter(0.93, 0.98, default=0.952, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_17 = DecimalParameter(-18.0, -10.0, default=-12.8, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_18 = DecimalParameter(16.0, 32.0, default=26.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_18 = DecimalParameter(0.98, 1.0, default=0.982, space='buy', decimals=3, optimize=False, load=True)

    buy_rsi_1h_min_19 = DecimalParameter(40.0, 70.0, default=50.0, space='buy', decimals=1, optimize=False, load=True)
    buy_chop_min_19 = DecimalParameter(20.0, 60.0, default=24.1, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_20 = DecimalParameter(20.0, 36.0, default=27.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_20 = DecimalParameter(14.0, 30.0, default=20.0, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_21 = DecimalParameter(10.0, 28.0, default=23.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_21 = DecimalParameter(18.0, 40.0, default=24.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_22 = DecimalParameter(0.5, 6.0, default=3.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_22 = DecimalParameter(0.98, 1.0, default=0.98, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_22 = DecimalParameter(0.93, 0.98, default=0.94, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_22 = DecimalParameter(2.0, 10.0, default=4.2, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_22 = DecimalParameter(26.0, 56.0, default=37.0, space='buy', decimals=1, optimize=False, load=True)

    buy_bb_offset_23 = DecimalParameter(0.97, 1.0, default=0.987, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_23 = DecimalParameter(2.0, 10.0, default=7.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_23 = DecimalParameter(20.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_23 = DecimalParameter(60.0, 80.0, default=70.0, space='buy', decimals=1, optimize=False, load=True)

    buy_24_rsi_max = DecimalParameter(26.0, 60.0, default=60.0, space='buy', decimals=1, optimize=False, load=True)
    buy_24_rsi_1h_min = DecimalParameter(40.0, 90.0, default=66.9, space='buy', decimals=1, optimize=False, load=True)

    is_optimize_slip = False
    max_slip = DecimalParameter(0.33, 1.00, default=0.33, decimals=3, optimize=is_optimize_slip , space='buy', load=True)

    buy_condition_12_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_12_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_12_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_12_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_12_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_12_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_12_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_12_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="24", space='buy', optimize=False, load=True)
    buy_12_protection__safe_dips            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="100", space='buy', optimize=False, load=True)
    buy_12_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_12_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    is_optimize_ewo_2 = False
    buy_rsi_fast_ewo_2 = IntParameter(15, 50, default=45, optimize = is_optimize_ewo_2)
    buy_rsi_ewo_2 = IntParameter(15, 50, default=35, optimize = is_optimize_ewo_2)
    buy_ema_low_2 = DecimalParameter(0.90, 1.2, default=0.970 , optimize = is_optimize_ewo_2)
    buy_ema_high_2 = DecimalParameter(0.90, 1.2, default=1.087 , optimize = is_optimize_ewo_2)
    buy_ewo_high_2 = DecimalParameter(2, 12, default=4.179, optimize = is_optimize_ewo_2)


    ## Sell params

    sell_btc_safe = IntParameter(-400, -300, default=-365, optimize = False)

    is_optimize_sell_stoploss = False
    sell_cmf = DecimalParameter(-0.4, 0.0, default=0.0, optimize = is_optimize_sell_stoploss)
    sell_ema_close_delta = DecimalParameter(0.022, 0.027, default= 0.024, optimize = is_optimize_sell_stoploss)
    sell_ema = DecimalParameter(0.97, 0.99, default=0.987 , optimize = is_optimize_sell_stoploss)

    is_optimize_deadfish = False
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0 , optimize = is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_deadfish)

    is_optimize_bleeding = False
    sell_bleeding_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_bleeding)
    sell_bleeding_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_bleeding)
    sell_bleeding_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_bleeding)

    is_optimize_cti_r = False
    sell_cti_r_cti = DecimalParameter(0.55, 1, default=0.5 , optimize = is_optimize_cti_r)
    sell_cti_r_r = DecimalParameter(-15, 0, default=-20 , optimize = is_optimize_cti_r)

    #DIAMOND
    is_optimize_diamondprofit = False
    sell_push = DecimalParameter(0, 2, decimals=3,  default=1 , optimize = is_optimize_diamondprofit)
    sell_shift = IntParameter(-10, 0, default=-6 , optimize = is_optimize_diamondprofit)
    sell_fast = IntParameter(2, 50, default=9 , optimize = is_optimize_diamondprofit)
    sell_slow = IntParameter(2, 50, default=18 , optimize = is_optimize_diamondprofit)

        # Sell

    sell_condition_1_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_5_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_6_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_7_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_8_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)

    # 48h for pump sell checks
    sell_pump_threshold_48_1 = DecimalParameter(0.5, 1.2, default=0.9, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_48_2 = DecimalParameter(0.4, 0.9, default=0.7, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_48_3 = DecimalParameter(0.3, 0.7, default=0.5, space='sell', decimals=2, optimize=False, load=True)

    # 36h for pump sell checks
    sell_pump_threshold_36_1 = DecimalParameter(0.5, 0.9, default=0.72, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_36_2 = DecimalParameter(3.0, 6.0, default=4.0, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_36_3 = DecimalParameter(0.8, 1.6, default=1.0, space='sell', decimals=2, optimize=False, load=True)

    # 24h for pump sell checks
    sell_pump_threshold_24_1 = DecimalParameter(0.5, 0.9, default=0.68, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_24_2 = DecimalParameter(0.3, 0.6, default=0.62, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_24_3 = DecimalParameter(0.2, 0.5, default=0.88, space='sell', decimals=2, optimize=False, load=True)

    sell_rsi_bb_1 = DecimalParameter(60.0, 80.0, default=79.5, space='sell', decimals=1, optimize=False, load=True)

    sell_rsi_bb_2 = DecimalParameter(72.0, 90.0, default=81, space='sell', decimals=1, optimize=False, load=True)

    sell_rsi_main_3 = DecimalParameter(77.0, 90.0, default=82, space='sell', decimals=1, optimize=False, load=True)

    sell_dual_rsi_rsi_4 = DecimalParameter(72.0, 84.0, default=73.4, space='sell', decimals=1, optimize=False, load=True)
    sell_dual_rsi_rsi_1h_4 = DecimalParameter(78.0, 92.0, default=79.6, space='sell', decimals=1, optimize=False, load=True)

    sell_ema_relative_5 = DecimalParameter(0.005, 0.05, default=0.024, space='sell', optimize=False, load=True)
    sell_rsi_diff_5 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)

    sell_rsi_under_6 = DecimalParameter(72.0, 90.0, default=79.0, space='sell', decimals=1, optimize=False, load=True)

    sell_rsi_1h_7 = DecimalParameter(80.0, 95.0, default=81.7, space='sell', decimals=1, optimize=False, load=True)

    sell_bb_relative_8 = DecimalParameter(1.05, 1.3, default=1.1, space='sell', decimals=3, optimize=False, load=True)

    sell_custom_profit_0 = DecimalParameter(0.01, 0.1, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_0 = DecimalParameter(30.0, 40.0, default=34.0, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_profit_1 = DecimalParameter(0.01, 0.1, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_1 = DecimalParameter(30.0, 50.0, default=35.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_2 = DecimalParameter(0.01, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_2 = DecimalParameter(30.0, 50.0, default=37.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_3 = DecimalParameter(0.01, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_3 = DecimalParameter(30.0, 50.0, default=42.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_4 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_4 = DecimalParameter(35.0, 50.0, default=43.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_5 = DecimalParameter(0.01, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_5 = DecimalParameter(35.0, 50.0, default=45.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_6 = DecimalParameter(0.01, 0.1, default=0.07, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_6 = DecimalParameter(38.0, 55.0, default=48.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_7 = DecimalParameter(0.01, 0.1, default=0.08, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_7 = DecimalParameter(40.0, 58.0, default=54.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_8 = DecimalParameter(0.06, 0.1, default=0.09, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_8 = DecimalParameter(40.0, 50.0, default=55.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_9 = DecimalParameter(0.05, 0.14, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_9 = DecimalParameter(40.0, 60.0, default=54.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_10 = DecimalParameter(0.1, 0.14, default=0.12, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_10 = DecimalParameter(38.0, 50.0, default=42.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_11 = DecimalParameter(0.16, 0.45, default=0.20, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_11 = DecimalParameter(28.0, 40.0, default=34.0, space='sell', decimals=2, optimize=False, load=True)

    # Profit under EMA200
    sell_custom_under_profit_0 = DecimalParameter(0.01, 0.4, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_0 = DecimalParameter(28.0, 40.0, default=35.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_1 = DecimalParameter(0.01, 0.10, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_1 = DecimalParameter(36.0, 60.0, default=56.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_2 = DecimalParameter(0.01, 0.10, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_2 = DecimalParameter(46.0, 66.0, default=57.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_3 = DecimalParameter(0.01, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_3 = DecimalParameter(50.0, 68.0, default=58.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_4 = DecimalParameter(0.02, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_4 = DecimalParameter(50.0, 68.0, default=59.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_5 = DecimalParameter(0.02, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_5 = DecimalParameter(46.0, 62.0, default=60.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_6 = DecimalParameter(0.03, 0.1, default=0.07, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_6 = DecimalParameter(44.0, 60.0, default=56.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_7 = DecimalParameter(0.04, 0.1, default=0.08, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_7 = DecimalParameter(46.0, 60.0, default=54.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_8 = DecimalParameter(0.06, 0.12, default=0.09, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_8 = DecimalParameter(40.0, 58.0, default=55.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_9 = DecimalParameter(0.08, 0.14, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_9 = DecimalParameter(40.0, 60.0, default=54.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_10 = DecimalParameter(0.1, 0.16, default=0.12, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_10 = DecimalParameter(30.0, 50.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_11 = DecimalParameter(0.16, 0.3, default=0.2, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_11 = DecimalParameter(24.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)

    # Profit targets for pumped pairs 48h 1
    sell_custom_pump_profit_1_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_1 = DecimalParameter(26.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_1_2 = DecimalParameter(0.01, 0.6, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_2 = DecimalParameter(36.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_1_3 = DecimalParameter(0.02, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_3 = DecimalParameter(38.0, 50.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_1_4 = DecimalParameter(0.06, 0.12, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_4 = DecimalParameter(36.0, 48.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_1_5 = DecimalParameter(0.14, 0.24, default=0.2, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_5 = DecimalParameter(20.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)

    # Profit targets for pumped pairs 36h 1
    sell_custom_pump_profit_2_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_1 = DecimalParameter(26.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_2_2 = DecimalParameter(0.01, 0.6, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_2 = DecimalParameter(36.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_2_3 = DecimalParameter(0.02, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_3 = DecimalParameter(38.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_2_4 = DecimalParameter(0.06, 0.12, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_4 = DecimalParameter(36.0, 48.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_2_5 = DecimalParameter(0.14, 0.24, default=0.2, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_5 = DecimalParameter(20.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)

    # Profit targets for pumped pairs 24h 1
    sell_custom_pump_profit_3_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_1 = DecimalParameter(26.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_3_2 = DecimalParameter(0.01, 0.6, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_2 = DecimalParameter(34.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_3_3 = DecimalParameter(0.02, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_3 = DecimalParameter(38.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_3_4 = DecimalParameter(0.06, 0.12, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_4 = DecimalParameter(36.0, 48.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_3_5 = DecimalParameter(0.14, 0.24, default=0.2, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_5 = DecimalParameter(20.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)

    # SMA descending
    sell_custom_dec_profit_min_1 = DecimalParameter(0.01, 0.10, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_dec_profit_max_1 = DecimalParameter(0.06, 0.16, default=0.12, space='sell', decimals=3, optimize=False, load=True)

    # Under EMA100
    sell_custom_dec_profit_min_2 = DecimalParameter(0.05, 0.12, default=0.07, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_dec_profit_max_2 = DecimalParameter(0.06, 0.2, default=0.16, space='sell', decimals=3, optimize=False, load=True)

    # Trail 1
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.2, default=0.16, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.4, 0.7, default=0.6, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.01, 0.08, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_rsi_min_1 = DecimalParameter(16.0, 36.0, default=20.0, space='sell', decimals=1, optimize=False, load=True)
    sell_trail_rsi_max_1 = DecimalParameter(30.0, 50.0, default=50.0, space='sell', decimals=1, optimize=False, load=True)

    # Trail 2
    sell_trail_profit_min_2 = DecimalParameter(0.08, 0.16, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.3, 0.5, default=0.4, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.02, 0.08, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_rsi_min_2 = DecimalParameter(16.0, 36.0, default=20.0, space='sell', decimals=1, optimize=False, load=True)
    sell_trail_rsi_max_2 = DecimalParameter(30.0, 50.0, default=50.0, space='sell', decimals=1, optimize=False, load=True)

    # Trail 3
    sell_trail_profit_min_3 = DecimalParameter(0.01, 0.12, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_3 = DecimalParameter(0.1, 0.3, default=0.2, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_3 = DecimalParameter(0.01, 0.06, default=0.05, space='sell', decimals=3, optimize=False, load=True)

    # Under & near EMA200, accept profit
    sell_custom_profit_under_rel_1 = DecimalParameter(0.01, 0.04, default=0.024, space='sell', optimize=False, load=True)
    sell_custom_profit_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)

    # Under & near EMA200, take the loss
    sell_custom_stoploss_under_rel_1 = DecimalParameter(0.001, 0.02, default=0.004, space='sell', optimize=False, load=True)
    sell_custom_stoploss_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=10.0, space='sell', optimize=False, load=True)

    # Long duration/recover stoploss 1
    sell_custom_stoploss_long_profit_min_1 = DecimalParameter(-0.1, -0.02, default=-0.08, space='sell', optimize=False, load=True)
    sell_custom_stoploss_long_profit_max_1 = DecimalParameter(-0.06, -0.01, default=-0.04, space='sell', optimize=False, load=True)
    sell_custom_stoploss_long_recover_1 = DecimalParameter(0.05, 0.15, default=0.1, space='sell', optimize=False, load=True)
    sell_custom_stoploss_long_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=4.0, space='sell', optimize=False, load=True)

    # Long duration/recover stoploss 2
    sell_custom_stoploss_long_recover_2 = DecimalParameter(0.03, 0.15, default=0.06, space='sell', optimize=False, load=True)
    sell_custom_stoploss_long_rsi_diff_2 = DecimalParameter(30.0, 50.0, default=40.0, space='sell', optimize=False, load=True)

    # Pumped, descending SMA
    sell_custom_pump_dec_profit_min_1 = DecimalParameter(0.001, 0.04, default=0.005, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_max_1 = DecimalParameter(0.03, 0.08, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_min_2 = DecimalParameter(0.01, 0.08, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_max_2 = DecimalParameter(0.04, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_min_3 = DecimalParameter(0.02, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_max_3 = DecimalParameter(0.06, 0.12, default=0.09, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_min_4 = DecimalParameter(0.01, 0.05, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_max_4 = DecimalParameter(0.02, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)

    # Pumped 48h 1, under EMA200
    sell_custom_pump_under_profit_min_1 = DecimalParameter(0.02, 0.06, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_under_profit_max_1 = DecimalParameter(0.04, 0.1, default=0.09, space='sell', decimals=3, optimize=False, load=True)

    # Pumped trail 1
    sell_custom_pump_trail_profit_min_1 = DecimalParameter(0.01, 0.12, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_trail_profit_max_1 = DecimalParameter(0.06, 0.16, default=0.07, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_pump_trail_down_1 = DecimalParameter(0.01, 0.06, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_trail_rsi_min_1 = DecimalParameter(16.0, 36.0, default=20.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_trail_rsi_max_1 = DecimalParameter(30.0, 50.0, default=70.0, space='sell', decimals=1, optimize=False, load=True)

    # Stoploss, pumped, 48h 1
    sell_custom_stoploss_pump_max_profit_1 = DecimalParameter(0.01, 0.04, default=0.025, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_min_1 = DecimalParameter(-0.1, -0.01, default=-0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_max_1 = DecimalParameter(-0.1, -0.01, default=-0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_ma_offset_1 = DecimalParameter(0.7, 0.99, default=0.94, space='sell', decimals=2, optimize=False, load=True)

    # Stoploss, pumped, 48h 1
    sell_custom_stoploss_pump_max_profit_2 = DecimalParameter(0.01, 0.04, default=0.025, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_loss_2 = DecimalParameter(-0.1, -0.01, default=-0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_ma_offset_2 = DecimalParameter(0.7, 0.99, default=0.92, space='sell', decimals=2, optimize=False, load=True)

    # Stoploss, pumped, 36h 3
    sell_custom_stoploss_pump_max_profit_3 = DecimalParameter(0.01, 0.04, default=0.008, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_loss_3 = DecimalParameter(-0.16, -0.06, default=-0.12, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_ma_offset_3 = DecimalParameter(0.7, 0.99, default=0.88, space='sell', decimals=2, optimize=False, load=True)

    # Recover
    sell_custom_recover_profit_1 = DecimalParameter(0.01, 0.06, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_recover_min_loss_1 = DecimalParameter(0.06, 0.16, default=0.12, space='sell', decimals=3, optimize=False, load=True)

    sell_custom_recover_profit_min_2 = DecimalParameter(0.01, 0.04, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_recover_profit_max_2 = DecimalParameter(0.02, 0.08, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_recover_min_loss_2 = DecimalParameter(0.04, 0.16, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_recover_rsi_2 = DecimalParameter(32.0, 52.0, default=46.0, space='sell', decimals=1, optimize=False, load=True)

    # Profit for long duration trades
    sell_custom_long_profit_min_1 = DecimalParameter(0.01, 0.04, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_long_profit_max_1 = DecimalParameter(0.02, 0.08, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_long_duration_min_1 = IntParameter(700, 2000, default=900, space='sell', optimize=False, load=True)

    #############################################################

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    #############################################################

    def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
        """
            Rolling Percentage Change Maximum across interval.

            :param dataframe: DataFrame The original OHLC dataframe
            :param method: High to Low / Open to Close
            :param length: int The length to look back
        """
        df = dataframe.copy()
        if method == 'HL':
            return ((df['high'].rolling(length).max() - df['low'].rolling(length).min()) / df['low'].rolling(length).min())
        elif method == 'OC':
            return ((df['open'].rolling(length).max() - df['close'].rolling(length).min()) / df['close'].rolling(length).min())
        else:
            raise ValueError(f"Method {method} not defined!")

    def top_percent_change(self, dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        df = dataframe.copy()
        if length == 0:
            return ((df['open'] - df['close']) / df['close'])
        else:
            return ((df['open'].rolling(length).max() - df['close']) / df['close'])

    def range_maxgap(self, dataframe: DataFrame, length: int) -> float:
        """
        Maximum Price Gap across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        df = dataframe.copy()
        return (df['open'].rolling(length).max() - df['close'].rolling(length).min())

    def range_maxgap_adjusted(self, dataframe: DataFrame, length: int, adjustment: float) -> float:
        """
        Maximum Price Gap across interval adjusted.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        :param adjustment: int The adjustment to be applied
        """
        return (self.range_maxgap(dataframe,length) / adjustment)

    def range_height(self, dataframe: DataFrame, length: int) -> float:
        """
        Current close distance to range bottom.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        df = dataframe.copy()
        return (df['close'] - df['close'].rolling(length).min())

    def safe_pump(self, dataframe: DataFrame, length: int, thresh: float, pull_thresh: float) -> bool:
        """
        Determine if entry after a pump is safe.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        :param thresh: int Maximum percentage change threshold
        :param pull_thresh: int Pullback from interval maximum threshold
        """
        df = dataframe.copy()
        return (df[f'oc_pct_change_{length}'] < thresh) | (self.range_maxgap_adjusted(df, length, pull_thresh) > self.range_height(df, length))

    def safe_dips(self, dataframe: DataFrame, thresh_0, thresh_2, thresh_12, thresh_144) -> bool:
        """
        Determine if dip is safe to enter.

        :param dataframe: DataFrame The original OHLC dataframe
        :param thresh_0: Threshold value for 0 length top pct change
        :param thresh_2: Threshold value for 2 length top pct change
        :param thresh_12: Threshold value for 12 length top pct change
        :param thresh_144: Threshold value for 144 length top pct change
        """
        return ((dataframe['tpct_change_0'] < thresh_0) &
                (dataframe['tpct_change_2'] < thresh_2) &
                (dataframe['tpct_change_12'] < thresh_12) &
                (dataframe['tpct_change_144'] < thresh_144))

    ############################################################################

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs.extend = [(pair, '15m') for pair in pairs]

        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."

        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        # RSI
        informative_1h['rsi_14'] = ta.RSI(informative_1h, timeperiod=14)

        # SMA
        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)

        informative_1h['sma_200_dec_20'] = informative_1h['sma_200'] < informative_1h['sma_200'].shift(20)
        informative_1h['sma_200_dec_24'] = informative_1h['sma_200'] < informative_1h['sma_200'].shift(24)

        # EMA
        informative_1h['ema_8'] = ta.EMA(informative_1h, timeperiod=8)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h['ema_20'] = ta.EMA(informative_1h, timeperiod=20)
        informative_1h['ema_26'] = ta.EMA(informative_1h, timeperiod=26)
        informative_1h['ema_12'] = ta.EMA(informative_1h, timeperiod=12)
        informative_1h['ema_25'] = ta.EMA(informative_1h, timeperiod=25)
        informative_1h['ema_35'] = ta.EMA(informative_1h, timeperiod=35)

        # CTI
        informative_1h['cti'] = pta.cti(informative_1h["close"], length=20)
        informative_1h['cti_40'] = pta.cti(informative_1h["close"], length=40)

        # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb20_2_low'] = bollinger['lower']
        informative_1h['bb20_2_mid'] = bollinger['mid']
        informative_1h['bb20_2_upp'] = bollinger['upper']
        informative_1h['bb20_width'] = ((informative_1h['bb20_2_upp'] - informative_1h['bb20_2_low']) / informative_1h['bb20_2_mid'])

        # CRSI (3, 2, 100)
        crsi_closechange = informative_1h['close'] / informative_1h['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        informative_1h['crsi'] =  (ta.RSI(informative_1h['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(informative_1h['close'], 100)) / 3

        # Williams %R
        informative_1h['r_96'] = williams_r(informative_1h, period=96)
        informative_1h['r_480'] = williams_r(informative_1h, period=480)

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband2'] = bollinger2['lower']
        informative_1h['bb_middleband2'] = bollinger2['mid']
        informative_1h['bb_upperband2'] = bollinger2['upper']
        informative_1h['bb_width'] = ((informative_1h['bb_upperband2'] - informative_1h['bb_lowerband2']) / informative_1h['bb_middleband2'])

        # ROC
        informative_1h['roc'] = ta.ROC(dataframe, timeperiod=9)

        # MOMDIV
        mom = momdiv(informative_1h)
        informative_1h['momdiv_buy'] = mom['momdiv_buy']
        informative_1h['momdiv_sell'] = mom['momdiv_sell']
        informative_1h['momdiv_coh'] = mom['momdiv_coh']
        informative_1h['momdiv_col'] = mom['momdiv_col']

        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # CMF
        informative_1h['cmf'] = chaikin_money_flow(informative_1h, 20)

        # Heikin Ashi
        inf_heikinashi = qtpylib.heikinashi(informative_1h)
        informative_1h['ha_close'] = inf_heikinashi['close']
        informative_1h['rocr'] = ta.ROCR(informative_1h['ha_close'], timeperiod=168)

        # T3 Average
        informative_1h['T3'] = T3(informative_1h)

        # Elliot
        informative_1h['EWO'] = EWO(informative_1h, 50, 200)

        # nfi 37
        informative_1h['hl_pct_change_5'] = range_percent_change(informative_1h, 'HL', 5)
        informative_1h['low_5'] = informative_1h['low'].shift().rolling(5).min()
        informative_1h['safe_dump_50'] = ((informative_1h['hl_pct_change_5'] < 0.66) | (informative_1h['close'] < informative_1h['low_5']) | (informative_1h['close'] > informative_1h['open']))

        # Pump protections
        informative_1h['hl_pct_change_48'] = self.range_percent_change(informative_1h, 'HL', 48)
        informative_1h['hl_pct_change_36'] = self.range_percent_change(informative_1h, 'HL', 36)
        informative_1h['hl_pct_change_24'] = self.range_percent_change(informative_1h, 'HL', 24)

        informative_1h['oc_pct_change_48'] = self.range_percent_change(informative_1h, 'OC', 48)
        informative_1h['oc_pct_change_36'] = self.range_percent_change(informative_1h, 'OC', 36)
        informative_1h['oc_pct_change_24'] = self.range_percent_change(informative_1h, 'OC', 24)

        informative_1h['safe_pump_24_10'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_10_24.value, self.buy_pump_pull_threshold_10_24.value)
        informative_1h['safe_pump_36_10'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_10_36.value, self.buy_pump_pull_threshold_10_36.value)
        informative_1h['safe_pump_48_10'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_10_48.value, self.buy_pump_pull_threshold_10_48.value)

        informative_1h['safe_pump_24_20'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_20_24.value, self.buy_pump_pull_threshold_20_24.value)
        informative_1h['safe_pump_36_20'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_20_36.value, self.buy_pump_pull_threshold_20_36.value)
        informative_1h['safe_pump_48_20'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_20_48.value, self.buy_pump_pull_threshold_20_48.value)

        informative_1h['safe_pump_24_30'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_30_24.value, self.buy_pump_pull_threshold_30_24.value)
        informative_1h['safe_pump_36_30'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_30_36.value, self.buy_pump_pull_threshold_30_36.value)
        informative_1h['safe_pump_48_30'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_30_48.value, self.buy_pump_pull_threshold_30_48.value)

        informative_1h['safe_pump_24_40'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_40_24.value, self.buy_pump_pull_threshold_40_24.value)
        informative_1h['safe_pump_36_40'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_40_36.value, self.buy_pump_pull_threshold_40_36.value)
        informative_1h['safe_pump_48_40'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_40_48.value, self.buy_pump_pull_threshold_40_48.value)

        informative_1h['safe_pump_24_50'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_50_24.value, self.buy_pump_pull_threshold_50_24.value)
        informative_1h['safe_pump_36_50'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_50_36.value, self.buy_pump_pull_threshold_50_36.value)
        informative_1h['safe_pump_48_50'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_50_48.value, self.buy_pump_pull_threshold_50_48.value)

        informative_1h['safe_pump_24_60'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_60_24.value, self.buy_pump_pull_threshold_60_24.value)
        informative_1h['safe_pump_36_60'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_60_36.value, self.buy_pump_pull_threshold_60_36.value)
        informative_1h['safe_pump_48_60'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_60_48.value, self.buy_pump_pull_threshold_60_48.value)

        informative_1h['safe_pump_24_70'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_70_24.value, self.buy_pump_pull_threshold_70_24.value)
        informative_1h['safe_pump_36_70'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_70_36.value, self.buy_pump_pull_threshold_70_36.value)
        informative_1h['safe_pump_48_70'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_70_48.value, self.buy_pump_pull_threshold_70_48.value)

        informative_1h['safe_pump_24_80'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_80_24.value, self.buy_pump_pull_threshold_80_24.value)
        informative_1h['safe_pump_36_80'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_80_36.value, self.buy_pump_pull_threshold_80_36.value)
        informative_1h['safe_pump_48_80'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_80_48.value, self.buy_pump_pull_threshold_80_48.value)

        informative_1h['safe_pump_24_90'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_90_24.value, self.buy_pump_pull_threshold_90_24.value)
        informative_1h['safe_pump_36_90'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_90_36.value, self.buy_pump_pull_threshold_90_36.value)
        informative_1h['safe_pump_48_90'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_90_48.value, self.buy_pump_pull_threshold_90_48.value)

        informative_1h['safe_pump_24_100'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_100_24.value, self.buy_pump_pull_threshold_100_24.value)
        informative_1h['safe_pump_36_100'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_100_36.value, self.buy_pump_pull_threshold_100_36.value)
        informative_1h['safe_pump_48_100'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_100_48.value, self.buy_pump_pull_threshold_100_48.value)

        informative_1h['safe_pump_24_110'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_110_24.value, self.buy_pump_pull_threshold_110_24.value)
        informative_1h['safe_pump_36_110'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_110_36.value, self.buy_pump_pull_threshold_110_36.value)
        informative_1h['safe_pump_48_110'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_110_48.value, self.buy_pump_pull_threshold_110_48.value)

        informative_1h['safe_pump_24_120'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_120_24.value, self.buy_pump_pull_threshold_120_24.value)
        informative_1h['safe_pump_36_120'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_120_36.value, self.buy_pump_pull_threshold_120_36.value)
        informative_1h['safe_pump_48_120'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_120_48.value, self.buy_pump_pull_threshold_120_48.value)

        informative_1h['sell_pump_48_1'] = (informative_1h['hl_pct_change_48'] > self.sell_pump_threshold_48_1.value)
        informative_1h['sell_pump_48_2'] = (informative_1h['hl_pct_change_48'] > self.sell_pump_threshold_48_2.value)
        informative_1h['sell_pump_48_3'] = (informative_1h['hl_pct_change_48'] > self.sell_pump_threshold_48_3.value)

        informative_1h['sell_pump_36_1'] = (informative_1h['hl_pct_change_36'] > self.sell_pump_threshold_36_1.value)
        informative_1h['sell_pump_36_2'] = (informative_1h['hl_pct_change_36'] > self.sell_pump_threshold_36_2.value)
        informative_1h['sell_pump_36_3'] = (informative_1h['hl_pct_change_36'] > self.sell_pump_threshold_36_3.value)

        informative_1h['sell_pump_24_1'] = (informative_1h['hl_pct_change_24'] > self.sell_pump_threshold_24_1.value)
        informative_1h['sell_pump_24_2'] = (informative_1h['hl_pct_change_24'] > self.sell_pump_threshold_24_2.value)
        informative_1h['sell_pump_24_3'] = (informative_1h['hl_pct_change_24'] > self.sell_pump_threshold_24_3.value)
        informative_1h['ema_fast'] = ta.EMA(informative_1h, timeperiod=20)
        informative_1h['ema_slow'] = ta.EMA(informative_1h, timeperiod=25)

        informative_1h['uptrend'] = (
            (informative_1h['ema_fast'] > informative_1h['ema_slow'])
        ).astype('int')
        informative_1h['ema_15'] = ta.EMA(informative_1h, timeperiod=15)

        return informative_1h

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe_15m)

        # RSI
        informative_15m['rsi_14'] = ta.RSI(informative_15m, timeperiod=14)

        # EMAs
        informative_15m['ema_12'] = ta.EMA(informative_15m, timeperiod=12)
        informative_15m['ema_16'] = ta.EMA(informative_15m, timeperiod=16)
        informative_15m['ema_20'] = ta.EMA(informative_15m, timeperiod=20)
        informative_15m['ema_25'] = ta.EMA(informative_15m, timeperiod=25)
        informative_15m['ema_26'] = ta.EMA(informative_15m, timeperiod=26)
        informative_15m['ema_50'] = ta.EMA(informative_15m, timeperiod=50)
        informative_15m['ema_100'] = ta.EMA(informative_15m, timeperiod=100)
        informative_15m['ema_200'] = ta.EMA(informative_15m, timeperiod=200)

        # SMA
        informative_15m['sma_15'] = ta.SMA(informative_15m, timeperiod=15)
        informative_15m['sma_30'] = ta.SMA(informative_15m, timeperiod=30)
        informative_15m['sma_200'] = ta.SMA(informative_15m, timeperiod=200)

        informative_15m['sma_200_dec_20'] = informative_15m['sma_200'] < informative_15m['sma_200'].shift(20)

                # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_15m), window=20, stds=2)
        informative_15m['bb20_2_low'] = bollinger['lower']
        informative_15m['bb20_2_mid'] = bollinger['mid']
        informative_15m['bb20_2_upp'] = bollinger['upper']

        # BB 40 - STD2
        bb_40_std2 = qtpylib.bollinger_bands(informative_15m['close'], window=40, stds=2)
        informative_15m['bb40_2_low'] = bb_40_std2['lower']
        informative_15m['bb40_2_mid'] = bb_40_std2['mid']
        informative_15m['bb40_2_delta'] = (bb_40_std2['mid'] - informative_15m['bb40_2_low']).abs()
        informative_15m['closedelta'] = (informative_15m['close'] - informative_15m['close'].shift()).abs()
        informative_15m['tail'] = (informative_15m['close'] - informative_15m['bb40_2_low']).abs()

        # CMF
        informative_15m['cmf'] = chaikin_money_flow(informative_15m, 20)

        # CTI
        informative_15m['cti'] = pta.cti(informative_15m["close"], length=20)

        # Williams %R
        informative_15m['r_14'] = williams_r(informative_15m, period=14)
        informative_15m['r_64'] = williams_r(informative_15m, period=64)
        informative_15m['r_96'] = williams_r(informative_15m, period=96)

        # EWO
        informative_15m['ewo'] = EWO(informative_15m, 50, 200)

        # CCI
        informative_15m['cci'] = ta.CCI(informative_15m, source='hlc3', timeperiod=20)

        # CRSI (3, 2, 100)
        crsi_closechange = informative_15m['close'] / informative_15m['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        informative_15m['crsi'] =  (ta.RSI(informative_15m['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(informative_15m['close'], 100)) / 3

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")

        return informative_15m

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)
        max_loss = ((trade.open_rate - trade.min_rate) / trade.min_rate)

        if (last_candle is not None):
            if (current_profit > self.sell_custom_profit_11.value) & (last_candle['rsi'] < self.sell_custom_rsi_11.value):
                return 'signal_profit_11'
            if (self.sell_custom_profit_11.value > current_profit > self.sell_custom_profit_10.value) & (last_candle['rsi'] < self.sell_custom_rsi_10.value):
                return 'signal_profit_10'
            if (self.sell_custom_profit_10.value > current_profit > self.sell_custom_profit_9.value) & (last_candle['rsi'] < self.sell_custom_rsi_9.value):
                return 'signal_profit_9'
            if (self.sell_custom_profit_9.value > current_profit > self.sell_custom_profit_8.value) & (last_candle['rsi'] < self.sell_custom_rsi_8.value):
                return 'signal_profit_8'
            if (self.sell_custom_profit_8.value > current_profit > self.sell_custom_profit_7.value) & (last_candle['rsi'] < self.sell_custom_rsi_7.value):
                return 'signal_profit_7'
            if (self.sell_custom_profit_7.value > current_profit > self.sell_custom_profit_6.value) & (last_candle['rsi'] < self.sell_custom_rsi_6.value):
                return 'signal_profit_6'
            if (self.sell_custom_profit_6.value > current_profit > self.sell_custom_profit_5.value) & (last_candle['rsi'] < self.sell_custom_rsi_5.value):
                return 'signal_profit_5'
            elif (self.sell_custom_profit_5.value > current_profit > self.sell_custom_profit_4.value) & (last_candle['rsi'] < self.sell_custom_rsi_4.value):
                return 'signal_profit_4'
            elif (self.sell_custom_profit_4.value > current_profit > self.sell_custom_profit_3.value) & (last_candle['rsi'] < self.sell_custom_rsi_3.value):
                return 'signal_profit_3'
            elif (self.sell_custom_profit_3.value > current_profit > self.sell_custom_profit_2.value) & (last_candle['rsi'] < self.sell_custom_rsi_2.value):
                return 'signal_profit_2'
            elif (self.sell_custom_profit_2.value > current_profit > self.sell_custom_profit_1.value) & (last_candle['rsi'] < self.sell_custom_rsi_1.value):
                return 'signal_profit_1'
            elif (self.sell_custom_profit_1.value > current_profit > self.sell_custom_profit_0.value) & (last_candle['rsi'] < self.sell_custom_rsi_0.value):
                return 'signal_profit_0'

            # check if close is under EMA200
            elif (current_profit > self.sell_custom_under_profit_11.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_11.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_11'
            elif (self.sell_custom_under_profit_11.value > current_profit > self.sell_custom_under_profit_10.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_10.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_10'
            elif (self.sell_custom_under_profit_10.value > current_profit > self.sell_custom_under_profit_9.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_9.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_9'
            elif (self.sell_custom_under_profit_9.value > current_profit > self.sell_custom_under_profit_8.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_8.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_8'
            elif (self.sell_custom_under_profit_8.value > current_profit > self.sell_custom_under_profit_7.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_7.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_7'
            elif (self.sell_custom_under_profit_7.value > current_profit > self.sell_custom_under_profit_6.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_6.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_6'
            elif (self.sell_custom_under_profit_6.value > current_profit > self.sell_custom_under_profit_5.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_5.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_5'
            elif (self.sell_custom_under_profit_5.value > current_profit > self.sell_custom_under_profit_4.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_4.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_4'
            elif (self.sell_custom_under_profit_4.value > current_profit > self.sell_custom_under_profit_3.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_3.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_3'
            elif (self.sell_custom_under_profit_3.value > current_profit > self.sell_custom_under_profit_2.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_2.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_2'
            elif (self.sell_custom_under_profit_2.value > current_profit > self.sell_custom_under_profit_1.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_1.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_1'
            elif (self.sell_custom_under_profit_1.value > current_profit > self.sell_custom_under_profit_0.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_0.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_0'

            # check if the pair is "pumped"

            elif (last_candle['sell_pump_48_1_1h']) & (current_profit > self.sell_custom_pump_profit_1_5.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_5.value):
                return 'signal_profit_p_1_5'
            elif (last_candle['sell_pump_48_1_1h']) & (self.sell_custom_pump_profit_1_5.value > current_profit > self.sell_custom_pump_profit_1_4.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_4.value):
                return 'signal_profit_p_1_4'
            elif (last_candle['sell_pump_48_1_1h']) & (self.sell_custom_pump_profit_1_4.value > current_profit > self.sell_custom_pump_profit_1_3.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_3.value):
                return 'signal_profit_p_1_3'
            elif (last_candle['sell_pump_48_1_1h']) & (self.sell_custom_pump_profit_1_3.value > current_profit > self.sell_custom_pump_profit_1_2.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_2.value):
                return 'signal_profit_p_1_2'
            elif (last_candle['sell_pump_48_1_1h']) & (self.sell_custom_pump_profit_1_2.value > current_profit > self.sell_custom_pump_profit_1_1.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_1.value):
                return 'signal_profit_p_1_1'

            elif (last_candle['sell_pump_36_1_1h']) & (current_profit > self.sell_custom_pump_profit_2_5.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_5.value):
                return 'signal_profit_p_2_5'
            elif (last_candle['sell_pump_36_1_1h']) & (self.sell_custom_pump_profit_2_5.value > current_profit > self.sell_custom_pump_profit_2_4.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_4.value):
                return 'signal_profit_p_2_4'
            elif (last_candle['sell_pump_36_1_1h']) & (self.sell_custom_pump_profit_2_4.value > current_profit > self.sell_custom_pump_profit_2_3.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_3.value):
                return 'signal_profit_p_2_3'
            elif (last_candle['sell_pump_36_1_1h']) & (self.sell_custom_pump_profit_2_3.value > current_profit > self.sell_custom_pump_profit_2_2.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_2.value):
                return 'signal_profit_p_2_2'
            elif (last_candle['sell_pump_36_1_1h']) & (self.sell_custom_pump_profit_2_2.value > current_profit > self.sell_custom_pump_profit_2_1.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_1.value):
                return 'signal_profit_p_2_1'

            elif (last_candle['sell_pump_24_1_1h']) & (current_profit > self.sell_custom_pump_profit_3_5.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_5.value):
                return 'signal_profit_p_3_5'
            elif (last_candle['sell_pump_24_1_1h']) & (self.sell_custom_pump_profit_3_5.value > current_profit > self.sell_custom_pump_profit_3_4.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_4.value):
                return 'signal_profit_p_3_4'
            elif (last_candle['sell_pump_24_1_1h']) & (self.sell_custom_pump_profit_3_4.value > current_profit > self.sell_custom_pump_profit_3_3.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_3.value):
                return 'signal_profit_p_3_3'
            elif (last_candle['sell_pump_24_1_1h']) & (self.sell_custom_pump_profit_3_3.value > current_profit > self.sell_custom_pump_profit_3_2.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_2.value):
                return 'signal_profit_p_3_2'
            elif (last_candle['sell_pump_24_1_1h']) & (self.sell_custom_pump_profit_3_2.value > current_profit > self.sell_custom_pump_profit_3_1.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_1.value):
                return 'signal_profit_p_3_1'

            elif (self.sell_custom_dec_profit_max_1.value > current_profit > self.sell_custom_dec_profit_min_1.value) & (last_candle['sma_200_dec_20']):
                return 'signal_profit_d_1'
            elif (self.sell_custom_dec_profit_max_2.value > current_profit > self.sell_custom_dec_profit_min_2.value) & (last_candle['close'] < last_candle['ema_100']):
                return 'signal_profit_d_2'

            # Trailing
            elif (self.sell_trail_profit_max_1.value > current_profit > self.sell_trail_profit_min_1.value) & (self.sell_trail_rsi_min_1.value < last_candle['rsi'] < self.sell_trail_rsi_max_1.value) & (max_profit > (current_profit + self.sell_trail_down_1.value)):
                return 'signal_profit_t_1'
            elif (self.sell_trail_profit_max_2.value > current_profit > self.sell_trail_profit_min_2.value) & (self.sell_trail_rsi_min_2.value < last_candle['rsi'] < self.sell_trail_rsi_max_2.value) & (max_profit > (current_profit + self.sell_trail_down_2.value)):
                return 'signal_profit_t_2'
            elif (self.sell_trail_profit_max_3.value > current_profit > self.sell_trail_profit_min_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)) & (last_candle['sma_200_dec_20_1h']):
                return 'signal_profit_t_3'

            elif (last_candle['close'] < last_candle['ema_200']) & (current_profit > self.sell_trail_profit_min_3.value) & (current_profit < self.sell_trail_profit_max_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)):
                return 'signal_profit_u_t_1'

            # elif (last_candle['sell_pump_48_1_1h']) & (0.06 > current_profit > 0.04) & (last_candle['rsi'] < 54.0) & (current_time - timedelta(minutes=30) < trade.open_date_utc):
            #     return 'signal_profit_p_s_1'

            elif (current_profit > 0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_profit_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_profit_under_rsi_diff_1.value):
                return 'signal_profit_u_e_1'

            elif (current_profit < -0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_stoploss_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_under_rsi_diff_1.value) & (last_candle['sma_200_dec_24']) & (current_time - timedelta(minutes=720) > trade.open_date_utc):
                return 'signal_stoploss_u_1'

            elif (self.sell_custom_stoploss_long_profit_min_1.value < current_profit < self.sell_custom_stoploss_long_profit_max_1.value) & (current_profit > (-max_loss + self.sell_custom_stoploss_long_recover_1.value)) & (last_candle['close'] < last_candle['ema_200'])  & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_long_rsi_diff_1.value) & (last_candle['sma_200_dec_24']) & (current_time - timedelta(minutes=1200) > trade.open_date_utc):
                return 'signal_stoploss_l_r_u_1'

            elif (current_profit < -0.0) & (current_profit > (-max_loss + self.sell_custom_stoploss_long_recover_2.value)) & (last_candle['close'] < last_candle['ema_200'])  & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_long_rsi_diff_2.value) & (last_candle['sma_200_dec_24']) & (current_time - timedelta(minutes=1200) > trade.open_date_utc):
                return 'signal_stoploss_l_r_u_2'

            elif (self.sell_custom_pump_dec_profit_max_1.value > current_profit > self.sell_custom_pump_dec_profit_min_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['sma_200_dec_20']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_d_1'
            elif (self.sell_custom_pump_dec_profit_max_2.value > current_profit > self.sell_custom_pump_dec_profit_min_2.value) & (last_candle['sell_pump_48_2_1h']) & (last_candle['sma_200_dec_20']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_d_2'
            elif (self.sell_custom_pump_dec_profit_max_3.value > current_profit > self.sell_custom_pump_dec_profit_min_3.value) & (last_candle['sell_pump_48_3_1h']) & (last_candle['sma_200_dec_20']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_d_3'
            elif (self.sell_custom_pump_dec_profit_max_4.value > current_profit > self.sell_custom_pump_dec_profit_min_4.value) & (last_candle['sma_200_dec_20']) & (last_candle['sell_pump_24_2_1h']):
                return 'signal_profit_p_d_4'

            # Pumped 48h 1, under EMA200
            elif (self.sell_custom_pump_under_profit_max_1.value > current_profit > self.sell_custom_pump_under_profit_min_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_u_1'

            # Pumped 36h 2, trail 1
            elif (last_candle['sell_pump_36_2_1h']) & (self.sell_custom_pump_trail_profit_max_1.value > current_profit > self.sell_custom_pump_trail_profit_min_1.value) & (self.sell_custom_pump_trail_rsi_min_1.value < last_candle['rsi'] < self.sell_custom_pump_trail_rsi_max_1.value) & (max_profit > (current_profit + self.sell_custom_pump_trail_down_1.value)):
                return 'signal_profit_p_t_1'

            # elif (max_profit < self.sell_custom_stoploss_pump_max_profit_1.value) & (self.sell_custom_stoploss_pump_min_1.value < current_profit < self.sell_custom_stoploss_pump_max_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['sma_200_dec_20']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_1.value)):
            #     return 'signal_stoploss_p_1'

            elif (max_profit < self.sell_custom_stoploss_pump_max_profit_2.value) & (current_profit < self.sell_custom_stoploss_pump_loss_2.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['sma_200_dec_20_1h']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_2.value)):
                return 'signal_stoploss_p_2'

            elif (max_profit < self.sell_custom_stoploss_pump_max_profit_3.value) & (current_profit < self.sell_custom_stoploss_pump_loss_3.value) & (last_candle['sell_pump_36_3_1h']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_3.value)):
                return 'signal_stoploss_p_3'

            # Recover
            elif (max_loss > self.sell_custom_recover_min_loss_1.value) & (current_profit > self.sell_custom_recover_profit_1.value):
                return 'signal_profit_r_1'

            elif (max_loss > self.sell_custom_recover_min_loss_2.value) & (self.sell_custom_recover_profit_max_2.value > current_profit > self.sell_custom_recover_profit_min_2.value) & (last_candle['rsi'] < self.sell_custom_recover_rsi_2.value):
                return 'signal_profit_r_2'

            # Take profit for long duration trades
            elif (self.sell_custom_long_profit_min_1.value < current_profit < self.sell_custom_long_profit_max_1.value) & (current_time - timedelta(minutes=self.sell_custom_long_duration_min_1.value) > trade.open_date_utc):
                return 'signal_profit_l_1'

        return None

    ## Confirm Entry
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        max_slip = self.max_slip.value

        if(len(dataframe) < 1):
            return False

        dataframe = dataframe.iloc[-1].squeeze()
        if ((rate > dataframe['close'])) :

            slippage = ( (rate / dataframe['close']) - 1 ) * 100

            if slippage < max_slip:
                return True
            else:
                return False

        return True

    ############################################################################

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # RSI
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_20'] = ta.RSI(dataframe, timeperiod=20)

        # Zero-Lag EMA
        dataframe['zema_61'] = zema(dataframe, period=61)
        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        # BB 40 - STD2
        bb_40_std2 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['bb40_2_low'] = bb_40_std2['lower']
        dataframe['bb40_2_mid'] = bb_40_std2['mid']
        dataframe['bb40_2_delta'] = (bb_40_std2['mid'] - dataframe['bb40_2_low']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['bb40_2_low']).abs()

        # BB 20 - STD2
        bb_20_std2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb20_2_low'] = bb_20_std2['lower']
        dataframe['bb20_2_mid'] = bb_20_std2['mid']
        dataframe['bb20_2_upp'] = bb_20_std2['upper']

        # BB 20 - STD3
        bb_20_std3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb20_3_low'] = bb_20_std3['lower']
        dataframe['bb20_3_mid'] = bb_20_std3['mid']
        dataframe['bb20_3_upp'] = bb_20_std3['upper']

        ### Other BB checks
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])

        # CCI hyperopt
        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cci'] = ta.CCI(dataframe, 26)
        dataframe['cci_long'] = ta.CCI(dataframe, 170)

        # RMI hyperopt
        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)

        # SRSI hyperopt
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        # SMA
        dataframe['bb9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_21'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['sma_28'] = ta.SMA(dataframe, timeperiod=28)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # CRSI (3, 2, 100)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] =  (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3

        # EMA
        dataframe['ema_4'] = ta.EMA(dataframe, timeperiod=4)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_25'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, 50, 200)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['r_32'] = williams_r(dataframe, period=32)
        dataframe['r_64'] = williams_r(dataframe, period=64)
        dataframe['r_96'] = williams_r(dataframe, period=96)
        dataframe['r_480'] = williams_r(dataframe, period=480)

        # Volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        ## BB 40
        bollinger2_40 = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        dataframe['bb_upperband2_40'] = bollinger2_40['upper']

        # ClucHA
        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

        # Profit Maximizer - PMAX
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)

        # MOMDIV
        mom = momdiv(dataframe)
        dataframe['momdiv_buy'] = mom['momdiv_buy']
        dataframe['momdiv_sell'] = mom['momdiv_sell']
        dataframe['momdiv_coh'] = mom['momdiv_coh']
        dataframe['momdiv_col'] = mom['momdiv_col']

        # T3 Average
        dataframe['T3'] = T3(dataframe)

        # True range
        dataframe['trange'] = ta.TRANGE(dataframe)

        # KC
        dataframe['range_ma_28'] = ta.SMA(dataframe['trange'], 28)
        dataframe['kc_upperband_28_1'] = dataframe['sma_28'] + dataframe['range_ma_28']
        dataframe['kc_lowerband_28_1'] = dataframe['sma_28'] - dataframe['range_ma_28']

        # KC 20
        dataframe['range_ma_20'] = ta.SMA(dataframe['trange'], 20)
        dataframe['kc_upperband_20_2'] = dataframe['sma_20'] + dataframe['range_ma_20'] * 2
        dataframe['kc_lowerband_20_2'] = dataframe['sma_20'] - dataframe['range_ma_20'] * 2
        dataframe['kc_bb_delta'] =  ( dataframe['kc_lowerband_20_2'] - dataframe['bb_lowerband2'] ) / dataframe['bb_lowerband2'] * 100

        # Linreg
        dataframe['hh_20'] = ta.MAX(dataframe['high'], 20)
        dataframe['ll_20'] = ta.MIN(dataframe['low'], 20)
        dataframe['avg_hh_ll_20'] = (dataframe['hh_20'] + dataframe['ll_20']) / 2
        dataframe['avg_close_20'] = ta.SMA(dataframe['close'], 20)
        dataframe['avg_val_20'] = (dataframe['avg_hh_ll_20'] + dataframe['avg_close_20']) / 2
        dataframe['linreg_val_20'] = ta.LINEARREG(dataframe['close'] - dataframe['avg_val_20'], 20, 0)

        # fisher
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # Modified Elder Ray Index
        dataframe['moderi_96'] = moderi(dataframe, 96)
        bb_40_std2 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['bb40_2_low']= bb_40_std2['lower']


        # EMA 200
        dataframe['ema_15'] = ta.EMA(dataframe, timeperiod=15)
        dataframe['ema_35'] = ta.EMA(dataframe, timeperiod=35)


        dataframe['ma_lower'] = ta.SMA(dataframe, timeperiod=15) * 0.953

        dataframe['rsi_slow_descending'] = (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift()).astype('int')
        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['sma_200_dec_20'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        dataframe['sma_200_dec_24'] = dataframe['sma_200'] < dataframe['sma_200'].shift(24)

        # Chopiness
        dataframe['chop']= qtpylib.chopiness(dataframe, 14)

        dataframe['zema'] = zema(dataframe, period=61)

        # Dip protection
        dataframe['tpct_change_0']   = self.top_percent_change(dataframe,0)
        dataframe['tpct_change_2']   = self.top_percent_change(dataframe,2)
        dataframe['tpct_change_12']  = self.top_percent_change(dataframe,12)
        dataframe['tpct_change_144'] = self.top_percent_change(dataframe,144)

        dataframe['safe_dips_10']  = self.safe_dips(dataframe, self.buy_dip_threshold_10_1.value, self.buy_dip_threshold_10_2.value, self.buy_dip_threshold_10_3.value, self.buy_dip_threshold_10_4.value)
        dataframe['safe_dips_20']  = self.safe_dips(dataframe, self.buy_dip_threshold_20_1.value, self.buy_dip_threshold_20_2.value, self.buy_dip_threshold_20_3.value, self.buy_dip_threshold_20_4.value)
        dataframe['safe_dips_30']  = self.safe_dips(dataframe, self.buy_dip_threshold_30_1.value, self.buy_dip_threshold_30_2.value, self.buy_dip_threshold_30_3.value, self.buy_dip_threshold_30_4.value)
        dataframe['safe_dips_40']  = self.safe_dips(dataframe, self.buy_dip_threshold_40_1.value, self.buy_dip_threshold_40_2.value, self.buy_dip_threshold_40_3.value, self.buy_dip_threshold_40_4.value)
        dataframe['safe_dips_50']  = self.safe_dips(dataframe, self.buy_dip_threshold_50_1.value, self.buy_dip_threshold_50_2.value, self.buy_dip_threshold_50_3.value, self.buy_dip_threshold_50_4.value)
        dataframe['safe_dips_60']  = self.safe_dips(dataframe, self.buy_dip_threshold_60_1.value, self.buy_dip_threshold_60_2.value, self.buy_dip_threshold_60_3.value, self.buy_dip_threshold_60_4.value)
        dataframe['safe_dips_70']  = self.safe_dips(dataframe, self.buy_dip_threshold_70_1.value, self.buy_dip_threshold_70_2.value, self.buy_dip_threshold_70_3.value, self.buy_dip_threshold_70_4.value)
        dataframe['safe_dips_80']  = self.safe_dips(dataframe, self.buy_dip_threshold_80_1.value, self.buy_dip_threshold_80_2.value, self.buy_dip_threshold_80_3.value, self.buy_dip_threshold_80_4.value)
        dataframe['safe_dips_90']  = self.safe_dips(dataframe, self.buy_dip_threshold_90_1.value, self.buy_dip_threshold_90_2.value, self.buy_dip_threshold_90_3.value, self.buy_dip_threshold_90_4.value)
        dataframe['safe_dips_100'] = self.safe_dips(dataframe, self.buy_dip_threshold_100_1.value, self.buy_dip_threshold_100_2.value, self.buy_dip_threshold_100_3.value, self.buy_dip_threshold_100_4.value)
        dataframe['safe_dips_110'] = self.safe_dips(dataframe, self.buy_dip_threshold_110_1.value, self.buy_dip_threshold_110_2.value, self.buy_dip_threshold_110_3.value, self.buy_dip_threshold_110_4.value)

        dataframe['volume_mean_30'] = dataframe['volume'].rolling(30).mean()

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        buy_12_protections = [True]
        if self.buy_12_protection__ema_fast.value:
            (
                buy_12_protections.append(dataframe[f"ema_{self.buy_12_protection__ema_fast_len.value}"] > dataframe['ema_200']) &
                buy_12_protections.append(dataframe[f"ema_{self.buy_12_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h']) &
                buy_12_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_12_protection__close_above_ema_fast_len.value}"]) &
                buy_12_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_12_protection__close_above_ema_slow_len.value}_1h"]) &
                buy_12_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_12_protection__sma200_rising_val.value))) &
                buy_12_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_12_protection__sma200_1h_rising_val.value))) &
                buy_12_protections.append(dataframe[f"safe_dips_{self.buy_12_protection__safe_dips_type.value}"]) &
                buy_12_protections.append(dataframe[f"safe_pump_{self.buy_12_protection__safe_pump_period.value}_{self.buy_12_protection__safe_pump_type.value}_1h"])
            )

        is_dip = (
                (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value)
            )

        is_sqzOff = (
                (dataframe['bb_lowerband2'] < dataframe['kc_lowerband_28_1']) &
                (dataframe['bb_upperband2'] > dataframe['kc_upperband_28_1'])
            )

        is_break = (

                (dataframe['bb_delta'] > self.buy_bb_delta.value) &
                (dataframe['bb_width'] > self.buy_bb_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &    # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value)
            )

        is_local_uptrend = (                                                                            # from NFI next gen, credit goes to @iterativ
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 )
            )

        is_local_dip = (
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff_local_dip.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ema_high_local_dip.value) &
                (dataframe['rsi'] < self.buy_rsi_local_dip.value) &
                (dataframe['crsi'] > self.buy_crsi_local_dip.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta_local_dip.value / 1000 )
            )

        is_ewo = (                                                                                      # from SMA offset
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                (dataframe['EWO'] > self.buy_ewo.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
            )

        is_ewo_2 = (
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_ewo_2.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low_2.value) &
                (dataframe['EWO'] > self.buy_ewo_high_2.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high_2.value) &
                (dataframe['rsi'] < self.buy_rsi_ewo_2.value)
            )

        is_nfix_3 = (
                (dataframe['bb40_2_low'].shift().gt(0)) &
                (dataframe['bb40_2_delta'].gt(dataframe['close'] * 0.045)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.02)) &
                (dataframe['tail'].lt(dataframe['bb40_2_delta'] * 0.24)) &
                (dataframe['close'].lt(dataframe['bb40_2_low'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['cti'] < -0.5) &
                (dataframe['r_14'] < -90.0) &
                (dataframe['r_96'] < -80.0) &
                (dataframe['cti_1h'] < -0.75) &
                (dataframe['r_480_1h'] < -30.0)
            )

        is_r_deadfish = (                                                                               # reverse deadfish
                (dataframe['ema_100'] < dataframe['ema_200'] * self.buy_r_deadfish_ema.value) &
                (dataframe['bb_width'] > self.buy_r_deadfish_bb_width.value) &
                (dataframe['close'] < dataframe['bb_middleband2'] * self.buy_r_deadfish_bb_factor.value) &
                (dataframe['volume_mean_12'] > dataframe['volume_mean_24'] * self.buy_r_deadfish_volume_factor.value) &
                (dataframe['cti'] < self.buy_r_deadfish_cti.value) &
                (dataframe['r_14'] < self.buy_r_deadfish_r14.value)
            )

        is_clucHA = (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h.value ) &
                (
                        (dataframe['bb_lowerband2_40'].shift() > 0) &
                        (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close.value) &
                        (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close.value) &
                        (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail.value) &
                        (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                        (dataframe['ha_close'] < dataframe['ha_close'].shift())
                )
            )

        is_cofi = (                                                                         # Modified from cofi, credit goes to original author "slack user CofiBit"
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value) &
                (dataframe['cti'] < self.buy_cofi_cti.value) &
                (dataframe['r_14'] < self.buy_cofi_r14.value)
            )

        is_gumbo = (                                                                        # Modified from gumbo1, creadit goes to original author @raph92
                (dataframe['EWO'] < self.buy_gumbo_ewo_low.value) &
                (dataframe['bb_middleband2_1h'] >= dataframe['T3_1h']) &
                (dataframe['T3'] <= dataframe['ema_8'] * self.buy_gumbo_ema.value) &
                (dataframe['cti'] < self.buy_gumbo_cti.value) &
                (dataframe['r_14'] < self.buy_gumbo_r14.value)
            )

        is_sqzmom = (                                                                       # Modified from squeezeMomentum, credit goes to original author @LazyBear of TradingView
                (is_sqzOff) &
                (dataframe['linreg_val_20'].shift(2) > dataframe['linreg_val_20'].shift(1)) &
                (dataframe['linreg_val_20'].shift(1) < dataframe['linreg_val_20']) &
                (dataframe['linreg_val_20'] < 0) &
                (dataframe['close'] < dataframe['ema_13'] * self.buy_sqzmom_ema.value) &
                (dataframe['EWO'] < self.buy_sqzmom_ewo.value) &
                (dataframe['r_14'] < self.buy_sqzmom_r14.value)
            )

        # NFI quick mode, credit goes to @iterativ
        is_nfi_13 = (
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['close'] < dataframe['sma_30'] * 0.99) &
                (dataframe['cti'] < -0.92) &
                (dataframe['EWO'] < -5.585) &
                (dataframe['cti_1h'] < -0.88) &
                (dataframe['crsi_1h'] > 10.0)
            )

        is_nfi_32 = (  # NFIX 26
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 46) &
                (dataframe['rsi'] > 25.0) &
                (dataframe['close'] < dataframe['sma_15'] * 0.93) &
                (dataframe['cti'] < -0.9)
            )

        is_nfi_33 = (
                (dataframe['close'] < (dataframe['ema_13'] * 0.978)) &
                (dataframe['EWO'] > 8) &
                (dataframe['cti'] < -0.88) &
                (dataframe['rsi'] < 32) &
                (dataframe['r_14'] < -98.0) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.5))
            )

        is_nfi_38 = (
                (dataframe['pm'] > dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * 0.98) &
                (dataframe['EWO'] < -4.4) &
                (dataframe['cti'] < -0.95) &
                (dataframe['r_14'] < -97) &
                (dataframe['crsi_1h'] > 0.5)
            )

        is_nfix_5 = (
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['close'] < dataframe['sma_75'] * 0.932) &
                (dataframe['EWO'] > 3.6) &
                (dataframe['cti'] < -0.9) &
                (dataframe['r_14'] < -97.0)
            )

        is_nfix_12 = (
                (dataframe['close'] < dataframe['ema_20'] * 0.932) &
                (dataframe['EWO'] > 0.1) &
                (dataframe['rsi_14'] < 40.0) &
                (dataframe['cti'] < -0.9) &
                (dataframe['r_480_1h'] < -20.0)
            )

        is_nfix_49 = (
                (dataframe['ema_26'].shift(3) > dataframe['ema_12'].shift(3)) &
                (dataframe['ema_26'].shift(3) - dataframe['ema_12'].shift(3) > dataframe['open'].shift(3) * 0.032) &
                (dataframe['ema_26'].shift(9) - dataframe['ema_12'].shift(9) > dataframe['open'].shift(3) / 100) &
                (dataframe['close'].shift(3) < dataframe['ema_20'].shift(3) * 0.916) &
                (dataframe['rsi'].shift(3) < 32.5) &
                (dataframe['crsi'].shift(3) > 18.0) &
                (dataframe['cti'] < self.buy_nfix_49_cti.value) &
                (dataframe['r_14'] < self.buy_nfix_49_r14.value)
            )

        is_nfi7_33 = (
                (dataframe['moderi_96']) &
                (dataframe['cti'] < -0.88) &
                (dataframe['close'] < (dataframe['ema_13'] * 0.988)) &
                (dataframe['EWO'] > 6.4) &
                (dataframe['rsi'] < 32.0) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.0))
            )

        is_nfi7_37 = (
                (dataframe['pm'] > dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * 0.98) &
                (dataframe['EWO'] > 9.8) &
                (dataframe['rsi'] < 56.0) &
                (dataframe['cti'] < -0.7) &
                (dataframe['safe_dump_50_1h'])
            )
            ## BB MODDED
        is_nfi_ctt35 = (
                (dataframe['pm'] <= dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * 0.984) &
                (dataframe['EWO'] > 9.6) &
                (dataframe['rsi_14'] < 32.0) &
                (dataframe['cti'] < -0.5)
            )

        is_nfi_ctt25 = (
                (dataframe['rsi_20'] < dataframe['rsi_20'].shift()) &
                (dataframe['rsi_4'] < self.buy_25_rsi_4.value) &
                (dataframe['ema_20_1h'] > dataframe['ema_26_1h']) &
                (dataframe['close'] < (dataframe['sma_20'] * self.buy_25_ma_offset.value)) &
                (dataframe['open'] > (dataframe['sma_20'] * self.buy_25_ma_offset.value)) &
                (
                    (dataframe['open'] < dataframe['ema_20_1h']) & (dataframe['low'] < dataframe['ema_20_1h']) |
                    (dataframe['open'] > dataframe['ema_20_1h']) & (dataframe['low'] > dataframe['ema_20_1h'])
                ) &
                (dataframe['cti'] < self.buy_25_cti.value)
            )

        is_nfi_ctt15 = (
                (dataframe['close'] > dataframe['ema_200_1h'] * self.buy_ema_rel_15.value) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_15.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['rsi_14'] < self.buy_rsi_15.value) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_15.value)
            )

        is_nfix_39 = (
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * 1.01)) &
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(48) * 1.07)) &
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['bb_lowerband2_40'].shift().gt(0)) &
                (dataframe['bb_delta_cluc'].gt(dataframe['close'] * 0.056)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.01)) &
                (dataframe['tail'].lt(dataframe['bb_delta_cluc'] * 0.5)) &
                (dataframe['close'].lt(dataframe['bb_lowerband2_40'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['close'] > dataframe['ema_13'] * self.buy_nfix_39_ema.value)
            )

        is_nfi_9 = (
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['close'] < dataframe['ema_20'] * 0.968) &
                (dataframe['close'] < dataframe['bb20_2_low'] * 0.982) &
                (dataframe['mfi'] < 50.0) &
                (dataframe['cti'] < -0.85) &
                (dataframe['r_14'] < -94.0) &
                (dataframe['rsi_14_1h'] > 20.0) &
                (dataframe['rsi_14_1h'] < 88.0) &
                (dataframe['crsi_1h'] > 21.0)
            )

        # EDIT ME MAYBE? this was ema fast = SMA now i made it ema_fast is larger then SMA
        is_diamond = (
                (dataframe['ema_20'] > ta.SMA(dataframe, timeperiod=int(self.buy_fast.value))) &
                (dataframe['ema_25'] > ta.SMA(dataframe, timeperiod=int(self.buy_slow.value)))
            )

        is_nfi_26 = (
                (dataframe['close'] < (dataframe['zema_61'] * 0.9405)) &
                (dataframe['cti'] < -0.72) &
                (dataframe['cci'] < -166.0) &
                (dataframe['r_14'] < -98.0) &
                (dataframe['cti_1h'] < 0.95) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.0))
            )

        is_nfix_54 = (
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * 1.01)) &
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(48) * 1.07)) &
                (dataframe['bb40_2_low'].shift().gt(0)) &
                (dataframe['bb40_2_delta'].gt(dataframe['close'] * 0.056)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.01)) &
                (dataframe['tail'].lt(dataframe['bb40_2_delta'] * 0.5)) &
                (dataframe['close'].lt(dataframe['bb40_2_low'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['close'] > dataframe['ema_50'] * 0.925)
            )

        is_nfix_53 = (
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['ema_200_1h'].shift(24) > dataframe['ema_200_1h'].shift(36)) &
                (dataframe['ema_26_15m'] > dataframe['ema_12_15m']) &
                ((dataframe['ema_26_15m'] - dataframe['ema_12_15m']) > (dataframe['open_15m'] * 0.02)) &
                ((dataframe['ema_26_15m'].shift(3) - dataframe['ema_12_15m'].shift(3)) > (dataframe['open_15m'] / 100)) &
                (dataframe['close_15m'] < (dataframe['bb20_2_low_15m'] * 0.99)) &
                (dataframe['r_14'] < -90.0) &
                (dataframe['cti_1h'] > -0.7)

            )
        is_nfix_52 = (
                (dataframe['ema_26_15m'] > dataframe['ema_12_15m']) &
                ((dataframe['ema_26_15m'] - dataframe['ema_12_15m']) > (dataframe['open_15m'] * 0.032)) &
                ((dataframe['ema_26_15m'].shift(3) - dataframe['ema_12_15m'].shift(3)) > (dataframe['open_15m'] / 100)) &
                (dataframe['close_15m'] < (dataframe['bb20_2_low_15m'] * 0.998)) &
                (dataframe['crsi_1h'] > 10.0)
            )

        is_nfix_51 = (
                (dataframe['close_15m'] < (dataframe['ema_16_15m'] * 0.944)) &
                (dataframe['ewo_15m'] < -1.0) &
                (dataframe['rsi_14_15m'] > 28.0) &
                (dataframe['cti_15m'] < -0.84) &
                (dataframe['r_14_15m'] < -94.0) &
                (dataframe['rsi_14'] > 30.0) &
                (dataframe['crsi_1h'] > 1.0)
            )

        is_nfix_48 = (
                (dataframe['close_15m'].shift(3) < (dataframe['sma_15_15m'].shift(3) * 0.95)) &
                (dataframe['close_15m'] > (dataframe['open_15m'].shift(3))) &
                (dataframe['ewo_15m'] > 2.8) &
                (dataframe['cti_15m'] < -0.75) &
                (dataframe['r_14_15m'].shift(3) < -94.0) &
                (dataframe['cti'] < -0.5) &
                (dataframe['cti_1h'] < 0.1) &
                (dataframe['crsi_1h'] > 18.0)
            )

        is_nfix_47 = (
                (dataframe['rsi_14_15m'] < dataframe['rsi_14_15m'].shift(3)) &
                (dataframe['ema_20_1h'] > dataframe['ema_25_1h']) &
                (dataframe['close_15m'] < (dataframe['sma_15_15m'] * 0.95)) &
                (
                    ((dataframe['open_15m'] < dataframe['ema_20_1h']) & (dataframe['low_15m'] < dataframe['ema_20_1h'])) |
                    ((dataframe['open_15m'] > dataframe['ema_20_1h']) & (dataframe['low_15m'] > dataframe['ema_20_1h']))
                ) &
                (dataframe['cti_15m'] < -0.9) &
                (dataframe['r_14_15m'] < -90.0) &
                (dataframe['r_14'] < -97.0) &
                (dataframe['cti_1h'] < 0.1) &
                (dataframe['crsi_1h'] > 8.0)
            )

        is_nfix_41 = (
                (dataframe['ema_12_15m'] > dataframe['ema_200_1h']) &
                (dataframe['ema_26_15m'] > dataframe['ema_12_15m']) &
                ((dataframe['ema_26_15m'] - dataframe['ema_12_15m']) > (dataframe['open_15m'] * 0.03)) &
                ((dataframe['ema_26_15m'].shift(3) - dataframe['ema_12_15m'].shift(3)) > (dataframe['open_15m'] / 100)) &
                (dataframe['close_15m'] < (dataframe['bb20_2_low_15m'] * 0.99))
            )

        is_nfix_38 = (
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * 1.01)) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.02)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['mfi'] < 34.5) &
                (dataframe['r_64'] < -65.0) &
                (dataframe['r_96'] < -50.0) &
                (dataframe['r_480_1h'] < -1.0)
            )

        is_nfix_36 = (
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(36) * 1.035)) &
                (dataframe['close'] < dataframe['ema_20'] * 0.956) &
                (dataframe['rsi_14'] < 34.0) &
                (dataframe['r_64'] < -80.0) &
                (dataframe['cti'] < -0.5) &
                (dataframe['r_480_1h'] < -30.0)
            )

        is_nfix_204 = (
                (dataframe['pm'] > dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * 0.98) &
                (dataframe['EWO'] < -4.4) &
                (dataframe['cti'] < -0.95) &
                (dataframe['r_14'] < -97.0) &
                (dataframe['crsi_1h'] > 0.5)
            )

        is_nfix_203 = (
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['close'] < dataframe['sma_30'] * 0.99) &
                (dataframe['cti'] < -0.92) &
                (dataframe['EWO'] < -6.0) &
                (dataframe['cti_1h'] < -0.88) &
                (dataframe['crsi_1h'] > 10.0)
            )

        is_nfix_202 = (
                (dataframe['close'] > (dataframe['ema_200_1h'] * 0.84)) &
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.02)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * 0.999)) &
                (dataframe['cti'] < -0.5) &
                (dataframe['rsi_14'] > 25.0) &
                (dataframe['mfi'] > 18.0) &
                (dataframe['r_14'] < -94.0) &
                (dataframe['r_14'].shift(1) < -94.0) &
                (dataframe['crsi_1h'] > 12.0) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 1.6))
            )

        is_nfix_201 = (
                (dataframe['rsi_20'] < dataframe['rsi_20'].shift()) &
                (dataframe['rsi_4'] < 30.0) &
                (dataframe['ema_20_1h'] > dataframe['ema_26_1h']) &
                (dataframe['close'] < dataframe['sma_15'] * 0.953) &
                (dataframe['cti'] < -0.78) &
                (dataframe['cci'] < -200.0)
            )

        is_nfix_34 = (
                (dataframe['close'] < dataframe['ema_50']) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * 0.972)) &
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi_14'] < 18.0)
            )

        is_nfix_28 = (
                (dataframe['close'] < dataframe['sma_75'] * 0.96) &
                (dataframe['EWO'] < -8.0) &
                (dataframe['cti'] < -0.9) &
                (dataframe['r_14'] < -97.0) &
                (dataframe['crsi_1h'] > 14.0)
            )

        is_nfix_27 = (
                (dataframe['close'] < dataframe['sma_75'] * 0.934) &
                (dataframe['EWO'] > 6.4) &
                (dataframe['rsi_14'] < 32.0) &
                (dataframe['cti'] < -0.8) &
                (dataframe['r_14'] < -96.0)
            )

        is_nfix_19 = (
                (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(12)) &
                (dataframe['ema_200_1h'].shift(12) > dataframe['ema_200_1h'].shift(24)) &
                (dataframe['bb40_2_low'].shift().gt(0)) &
                (dataframe['bb40_2_delta'].gt(dataframe['close'] * 0.045)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.02)) &
                (dataframe['tail'].lt(dataframe['bb40_2_delta'] * 0.28)) &
                (dataframe['close'].lt(dataframe['bb40_2_low'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['cti'] < -0.9) &
                (dataframe['cti_1h'] > -0.75) &
                (dataframe['cti_1h'] < 0.25)
            )

        is_nfix_11 = (
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.027)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < dataframe['ema_20'] * 0.932) &
                (dataframe['rsi_14'] < 25.0)
            )

        is_nfix_9 = (
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['close'] < dataframe['sma_30'] * 0.99) &
                (dataframe['cti'] < -0.92) &
                (dataframe['EWO'] < -5.0) &
                (dataframe['cti_1h'] < -0.88) &
                (dataframe['crsi_1h'] > 20.0)
            )

        is_nfix_36 = (
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(36) * 1.035)) &
                (dataframe['close'] < dataframe['ema_20'] * 0.956) &
                (dataframe['rsi_14'] < 34.0) &
                (dataframe['r_64'] < -80.0) &
                (dataframe['cti'] < -0.5) &
                (dataframe['r_480_1h'] < -30.0)
            )

        is_nfi_sma_1 = (
                (((dataframe['close'] - dataframe['open'].rolling(36).min()) / dataframe['open'].rolling(36).min()) > self.buy_min_inc_1.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_1.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_1.value) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_2 = (
                (dataframe['rsi'] < dataframe['rsi_1h'] - self.buy_rsi_1h_diff_2.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_2.value)) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_3 = (

                (dataframe['bb40_2_low'].shift().gt(0)) &
                (dataframe['bb40_2_delta'].gt(dataframe['close'] * self.buy_bb40_bbdelta_close_3.value)) &
                (dataframe['closedelta'].gt(dataframe['close'] * self.buy_bb40_closedelta_close_3.value)) &
                (dataframe['tail'].lt(dataframe['bb40_2_delta'] * self.buy_bb40_tail_bbdelta_3.value)) &
                (dataframe['close'].lt(dataframe['bb40_2_low'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_4 = (

                (dataframe['close'] < dataframe['ema_50']) &
                (dataframe['close'] < self.buy_bb20_close_bblowerband_4.value * dataframe['bb20_2_low']) &
                (dataframe['volume'] < (dataframe['volume_mean_30'].shift(1) * self.buy_bb20_volume_4.value))
            )

        is_nfi_sma_5 = (

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_5.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_5.value)) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_6 = (

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_6.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_6.value)) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_7 = (

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_7.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['rsi'] < self.buy_rsi_7.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_8 = (

                (dataframe['rsi'] < self.buy_rsi_8.value) &
                (dataframe['volume'] > (dataframe['volume'].shift(1) * self.buy_volume_8.value)) &
                (dataframe['close'] > dataframe['open']) &
                ((dataframe['close'] - dataframe['low']) > ((dataframe['close'] - dataframe['open']) * self.buy_tail_diff_8.value)) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_9 = (

                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_9.value) &
                (dataframe['close'] < dataframe['bb20_2_low'] * self.buy_bb_offset_9.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_9.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_9.value) &
                (dataframe['mfi'] < self.buy_mfi_9.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_10 = (

                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_10.value) &
                (dataframe['close'] < dataframe['bb20_2_low'] * self.buy_bb_offset_10.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_10.value) &
                (dataframe['volume'] > 0)
        )
        is_nfi_sma_11 = (

                (((dataframe['close'] - dataframe['open'].rolling(36).min()) / dataframe['open'].rolling(36).min()) > self.buy_min_inc_11.value) &
                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_11.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_11.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_11.value) &
                (dataframe['rsi'] < self.buy_rsi_11.value) &
                (dataframe['mfi'] < self.buy_mfi_11.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_12 = (

                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_12.value) &
                (dataframe['EWO'] > self.buy_ewo_12.value) &
                (dataframe['rsi'] < self.buy_rsi_12.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_13 = (

                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_13.value) &
                (dataframe['EWO'] < self.buy_ewo_13.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_14 = (

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_14.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_14.value)) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_14.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_15 = (

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_15.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['rsi'] < self.buy_rsi_15.value) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_15.value) &
                (dataframe['volume'] > 0)
                   )

        is_nfi_sma_16 = (

                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_16.value) &
                (dataframe['EWO'] > self.buy_ewo_16.value) &
                (dataframe['rsi'] < self.buy_rsi_16.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_17 = (

                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_17.value) &
                (dataframe['EWO'] < self.buy_ewo_17.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_18 = (

                (dataframe['rsi'] < self.buy_rsi_18.value) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_18.value)) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_19 = (

                (dataframe['close'].shift(1) > dataframe['ema_100_1h']) &
                (dataframe['low'] < dataframe['ema_100_1h']) &
                (dataframe['close'] > dataframe['ema_100_1h']) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_19.value) &
                (dataframe['chop'] < self.buy_chop_min_19.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_20 = (

                (dataframe['rsi'] < self.buy_rsi_20.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_20.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_21 = (

                (dataframe['rsi'] < self.buy_rsi_21.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_21.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_22 = (

                ((dataframe['volume_mean_4'] * self.buy_volume_22.value) > dataframe['volume']) &
                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_22.value) &
                (dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_22.value)) &
                (dataframe['EWO'] > self.buy_ewo_22.value) &
                (dataframe['rsi'] < self.buy_rsi_22.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_23 = (

                (dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_23.value)) &
                (dataframe['EWO'] > self.buy_ewo_23.value) &
                (dataframe['rsi'] < self.buy_rsi_23.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_23.value) &
                (dataframe['volume'] > 0)
            )

        is_nfi_sma_24 = (

                (dataframe['ema_12_1h'].shift(12) < dataframe['ema_35_1h'].shift(12)) &
                (dataframe['ema_12_1h'].shift(12) < dataframe['ema_35_1h'].shift(12)) &
                (dataframe['ema_12_1h'] > dataframe['ema_35_1h']) &
                (dataframe['cmf_1h'].shift(12) < 0) &
                (dataframe['cmf_1h'] > 0) &
                (dataframe['rsi'] < self.buy_24_rsi_max.value) &
                (dataframe['rsi_1h'] > self.buy_24_rsi_1h_min.value) &
                (dataframe['volume'] > 0)
            )

        is_additional_check = (
                (dataframe['roc_1h'] < self.buy_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value)
            )

        ## Additional Check
        is_BB_checked = is_dip & is_break

        ## Condition Append
        conditions.append(is_BB_checked)                                           # ~2.32 / 91.1% / 46.27%      D
        dataframe.loc[is_BB_checked, 'buy_tag'] += 'bb '

        conditions.append(is_local_uptrend)                                        # ~3.28 / 92.4% / 69.72%
        dataframe.loc[is_local_uptrend, 'buy_tag'] += 'local_uptrend '

        conditions.append(is_local_dip)                                            # ~0.76 / 91.1% / 15.54%
        dataframe.loc[is_local_dip, 'buy_tag'] += 'local_dip '

        conditions.append(is_ewo)                                                  # ~0.92 / 92.0% / 43.74%      D
        dataframe.loc[is_ewo, 'buy_tag'] += 'ewo '

        conditions.append(is_ewo_2)                                                  # ~0.92 / 92.0% / 43.74%      D
        dataframe.loc[is_ewo_2, 'buy_tag'] += 'ewo2 '

        conditions.append(is_nfix_3)                                                 # ~2.86 / 91.5% / 33.31%     D
        dataframe.loc[is_nfix_3, 'buy_tag'] += 'is_nfix_3 '

        conditions.append(is_r_deadfish)                                           # ~0.99 / 86.9% / 21.93%      D
        dataframe.loc[is_r_deadfish, 'buy_tag'] += 'r_deadfish '

        conditions.append(is_clucHA)                                               # ~7.2 / 92.5% / 97.98%       D
        dataframe.loc[is_clucHA, 'buy_tag'] += 'clucHA '

        conditions.append(is_cofi)                                                 # ~0.4 / 94.4% / 9.59%        D
        dataframe.loc[is_cofi, 'buy_tag'] += 'cofi '

        conditions.append(is_gumbo)                                                # ~2.63 / 90.6% / 41.49%      D
        dataframe.loc[is_gumbo, 'buy_tag'] += 'gumbo '

        conditions.append(is_sqzmom)                                               # ~3.14 / 92.4% / 64.14%      D
        dataframe.loc[is_sqzmom, 'buy_tag'] += 'sqzmom '

        conditions.append(is_nfi_13)                                               # ~0.4 / 100%                 D
        dataframe.loc[is_nfi_13, 'buy_tag'] += 'nfi_13 '

        conditions.append(is_nfi_32)                                               # ~0.78 / 92.0 % / 37.41%     D
        dataframe.loc[is_nfi_32, 'buy_tag'] += 'nfi_32 '

        conditions.append(is_nfi_33)                                               # ~0.11 / 100%                D
        dataframe.loc[is_nfi_33, 'buy_tag'] += 'nfi_33 '

        conditions.append(is_nfi_38)                                               # ~1.13 / 88.5% / 31.34%      D
        dataframe.loc[is_nfi_38, 'buy_tag'] += 'nfi_38 '

        conditions.append(is_nfix_5)                                               # ~0.25 / 97.7% / 6.53%       D
        dataframe.loc[is_nfix_5, 'buy_tag'] += 'nfix_5 '

        conditions.append(is_nfix_12)                                              # ~5.33 / 91.8% / 58.57%      D
        dataframe.loc[is_nfix_12, 'buy_tag'] += 'nfix_2 '

        conditions.append(is_nfix_49)                                              # ~0.33 / 100% / 0%           D
        dataframe.loc[is_nfix_49, 'buy_tag'] += 'nfix_49 '

        conditions.append(is_nfi7_33)                                              # ~0.71 / 91.3% / 28.94%      D
        dataframe.loc[is_nfi7_33, 'buy_tag'] += 'nfi7_33 '

        conditions.append(is_nfi7_37)                                              # ~0.46 / 92.6% / 17.05%      D
        dataframe.loc[is_nfi7_37, 'buy_tag'] += 'nfi7_37 '

        conditions.append(is_nfi_ctt35)                                           # ~2.32 / 91.1% / 46.27%      D
        dataframe.loc[is_nfi_ctt35, 'buy_tag'] += 'nfi_ctt35 '

        conditions.append(is_nfi_ctt25)                                        # ~3.28 / 92.4% / 69.72%
        dataframe.loc[is_nfi_ctt25, 'buy_tag'] += 'nfi_ctt25 '

        conditions.append(is_nfi_ctt15)                                            # ~0.76 / 91.1% / 15.54%
        dataframe.loc[is_nfi_ctt15, 'buy_tag'] += 'nfi_ctt15 '

        conditions.append(is_diamond)                                                  # ~0.92 / 92.0% / 43.74%      D
        dataframe.loc[is_diamond, 'buy_tag'] += 'diamond '

        conditions.append(is_nfix_54)                                           # ~0.99 / 86.9% / 21.93%      D
        dataframe.loc[is_nfix_54, 'buy_tag'] += 'nfix_54 '

        conditions.append(is_nfix_53)                                               # ~7.2 / 92.5% / 97.98%       D
        dataframe.loc[is_nfix_53, 'buy_tag'] += 'nfix_53 '

        conditions.append(is_nfix_52)                                                 # ~0.4 / 94.4% / 9.59%        D
        dataframe.loc[is_nfix_52, 'buy_tag'] += 'nfix_52 '

        conditions.append(is_nfix_51)                                                # ~2.63 / 90.6% / 41.49%      D
        dataframe.loc[is_nfix_51, 'buy_tag'] += 'nfix_51 '

        conditions.append(is_nfix_48)                                               # ~3.14 / 92.4% / 64.14%      D
        dataframe.loc[is_nfix_48, 'buy_tag'] += 'nfix_48 '

        conditions.append(is_nfix_47)                                               # ~0.4 / 100%                 D
        dataframe.loc[is_nfix_47, 'buy_tag'] += 'nfix_47 '

        conditions.append(is_nfix_41)                                               # ~0.78 / 92.0 % / 37.41%     D
        dataframe.loc[is_nfix_41, 'buy_tag'] += 'nfix_41 '

        conditions.append(is_nfix_38)                                               # ~0.11 / 100%                D
        dataframe.loc[is_nfix_38, 'buy_tag'] += 'nfix_38 '

        conditions.append(is_nfix_36)                                               # ~1.13 / 88.5% / 31.34%      D
        dataframe.loc[is_nfix_36, 'buy_tag'] += 'nfix_36 '

        conditions.append(is_nfix_204)                                               # ~0.25 / 97.7% / 6.53%       D
        dataframe.loc[is_nfix_204, 'buy_tag'] += 'nfix_204 '

        conditions.append(is_nfix_203)                                              # ~5.33 / 91.8% / 58.57%      D
        dataframe.loc[is_nfix_203, 'buy_tag'] += 'nfix_203 '

        conditions.append(is_nfix_202)                                              # ~0.33 / 100% / 0%           D
        dataframe.loc[is_nfix_202, 'buy_tag'] += 'nfix_202 '

        conditions.append(is_nfix_201)                                              # ~0.71 / 91.3% / 28.94%      D
        dataframe.loc[is_nfix_201, 'buy_tag'] += 'nfix_201 '

        conditions.append(is_nfix_34)                                              # ~0.46 / 92.6% / 17.05%      D
        dataframe.loc[is_nfix_34, 'buy_tag'] += 'nfix_34 '

        conditions.append(is_nfix_28)                                               # ~0.25 / 97.7% / 6.53%       D
        dataframe.loc[is_nfix_28, 'buy_tag'] += 'nfix_28 '

        conditions.append(is_nfix_27)                                              # ~5.33 / 91.8% / 58.57%      D
        dataframe.loc[is_nfix_27, 'buy_tag'] += 'nfix_27 '

        conditions.append(is_nfix_19)                                              # ~0.33 / 100% / 0%           D
        dataframe.loc[is_nfix_19, 'buy_tag'] += 'nfix_19 '

        conditions.append(is_nfix_11)                                              # ~0.71 / 91.3% / 28.94%      D
        dataframe.loc[is_nfix_11, 'buy_tag'] += 'nfix_11 '

        conditions.append(is_nfix_9)                                              # ~0.46 / 92.6% / 17.05%      D
        dataframe.loc[is_nfix_9, 'buy_tag'] += 'nfix_9 '

        conditions.append(is_nfix_36)                                              # ~0.46 / 92.6% / 17.05%      D
        dataframe.loc[is_nfix_36, 'buy_tag'] += 'nfix_36 '

        conditions.append(is_nfix_39)                                              # ~0.46 / 92.6% / 17.05%      D
        dataframe.loc[is_nfix_39, 'buy_tag'] += 'nfix_39 '

        conditions.append(is_nfi_sma_1)
        dataframe.loc[is_nfi_sma_1, 'buy_tag'] += 'is_nfi_sma_1 '

        conditions.append(is_nfi_sma_2)
        dataframe.loc[is_nfi_sma_2, 'buy_tag'] += 'is_nfi_sma_2 '

        conditions.append(is_nfi_sma_3)
        dataframe.loc[is_nfi_sma_3, 'buy_tag'] += 'is_nfi_sma_3 '

        conditions.append(is_nfi_sma_4)
        dataframe.loc[is_nfi_sma_4, 'buy_tag'] += 'is_nfi_sma_4 '

        conditions.append(is_nfi_sma_5)
        dataframe.loc[is_nfi_sma_5, 'buy_tag'] += 'is_nfi_sma_5 '

        conditions.append(is_nfi_sma_6)
        dataframe.loc[is_nfi_sma_6, 'buy_tag'] += 'is_nfi_sma_6 '

        conditions.append(is_nfi_sma_7)
        dataframe.loc[is_nfi_sma_7, 'buy_tag'] += 'is_nfi_sma_7 '

        conditions.append(is_nfi_sma_8)
        dataframe.loc[is_nfi_sma_8, 'buy_tag'] += 'is_nfi_sma_8 '

        conditions.append(is_nfi_sma_9)
        dataframe.loc[is_nfi_sma_9, 'buy_tag'] += 'is_nfi_sma_9 '

        conditions.append(is_nfi_sma_10)
        dataframe.loc[is_nfi_sma_10, 'buy_tag'] += 'is_nfi_sma_10 '

        conditions.append(is_nfi_sma_11)
        dataframe.loc[is_nfi_sma_11, 'buy_tag'] += 'is_nfi_sma_11 '

        conditions.append(is_nfi_sma_12)
        dataframe.loc[is_nfi_sma_12, 'buy_tag'] += 'is_nfi_sma_12 '

        conditions.append(is_nfi_sma_13)
        dataframe.loc[is_nfi_sma_13, 'buy_tag'] += 'is_nfi_sma_13 '

        conditions.append(is_nfi_sma_14)
        dataframe.loc[is_nfi_sma_14, 'buy_tag'] += 'is_nfi_sma_14 '

        conditions.append(is_nfi_sma_15)
        dataframe.loc[is_nfi_sma_15, 'buy_tag'] += 'is_nfi_sma_15 '

        conditions.append(is_nfi_sma_16)
        dataframe.loc[is_nfi_sma_16, 'buy_tag'] += 'is_nfi_sma_16 '

        conditions.append(is_nfi_sma_17)
        dataframe.loc[is_nfi_sma_17, 'buy_tag'] += 'is_nfi_sma_17 '

        conditions.append(is_nfi_sma_18)
        dataframe.loc[is_nfi_sma_18, 'buy_tag'] += 'is_nfi_sma_18 '

        conditions.append(is_nfi_sma_19)
        dataframe.loc[is_nfi_sma_19, 'buy_tag'] += 'is_nfi_sma_19 '

        conditions.append(is_nfi_sma_20)
        dataframe.loc[is_nfi_sma_20, 'buy_tag'] += 'is_nfi_sma_20 '

        conditions.append(is_nfi_sma_21)
        dataframe.loc[is_nfi_sma_21, 'buy_tag'] += 'is_nfi_sma_21 '

        conditions.append(is_nfi_sma_22)
        dataframe.loc[is_nfi_sma_22, 'buy_tag'] += 'is_nfi_sma_22 '

        conditions.append(is_nfi_sma_23)
        dataframe.loc[is_nfi_sma_23, 'buy_tag'] += 'is_nfi_sma_23 '

        conditions.append(is_nfi_sma_24)
        dataframe.loc[is_nfi_sma_24, 'buy_tag'] += 'is_nfi_sma_24 '

        if conditions:
            dataframe.loc[
                            is_additional_check
                            &
                            reduce(lambda x, y: x | y, conditions)

                        , 'buy' ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                self.sell_condition_1_enable.value &

                (dataframe['rsi'] > self.sell_rsi_bb_1.value) &
                (dataframe['close'] > dataframe['bb20_2_upp']) &
                (dataframe['close'].shift(1) > dataframe['bb20_2_upp'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb20_2_upp'].shift(2)) &
                (dataframe['close'].shift(3) > dataframe['bb20_2_upp'].shift(3)) &
                (dataframe['close'].shift(4) > dataframe['bb20_2_upp'].shift(4)) &
                (dataframe['close'].shift(5) > dataframe['bb20_2_upp'].shift(5)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_2_enable.value &

                (dataframe['rsi'] > self.sell_rsi_bb_2.value) &
                (dataframe['close'] > dataframe['bb20_2_upp']) &
                (dataframe['close'].shift(1) > dataframe['bb20_2_upp'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb20_2_upp'].shift(2)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_3_enable.value &

                (dataframe['rsi'] > self.sell_rsi_main_3.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_4_enable.value &

                (dataframe['rsi'] > self.sell_dual_rsi_rsi_4.value) &
                (dataframe['rsi_1h'] > self.sell_dual_rsi_rsi_1h_4.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_6_enable.value &

                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['rsi'] > self.sell_rsi_under_6.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_7_enable.value &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_7.value) &
                qtpylib.crossed_below(dataframe['ema_12'], dataframe['ema_26']) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_8_enable.value &

                (dataframe['close'] > dataframe['bb20_2_upp_1h'] * self.sell_bb_relative_8.value) &

                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        return dataframe

class BB_RPB_TSL_Tranz_TrailingBuy(BB_RPB_TSL_SMA_Tranz_TB_1_1_1):
    # Original idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal and will move to next candles if the trailing still active
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #

    process_only_new_candles = False

    custom_info_trail_buy = dict()

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 1800

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.02  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.000  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_trailing': False,
    }

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if (reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]):
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_dict.copy()
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            log.info(
                f"pair: {pair} : "
                f"start: {trailing_buy['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_profit_ratio(pair, current_price)*100:.2f}%, "
                f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        default_offset = 0.005

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if ((current_trailing_profit_ratio > 0) and (last_candle['buy'] == 1)):
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    # end of trailing buy parameters
    # -----------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)

        if val:
            if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if(len(dataframe) >= 1):
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_buy = self.trailing_buy(pair)
                    trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)

                    if trailing_buy['allow_trailing']:
                        if (not trailing_buy['trailing_buy_order_started'] and (last_candle['buy'] == 1)):
                            # start trailing buy

                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_started'] = True
                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_price'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['buy_tag'] = f"initial_buy_tag (strat trail price {last_candle['close']})"
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_time'] = datetime.now(timezone.utc)
                            # self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = 0

                            trailing_buy['trailing_buy_order_started'] = True
                            trailing_buy['trailing_buy_order_uplimit'] = last_candle['close']
                            trailing_buy['start_trailing_price'] = last_candle['close']
                            trailing_buy['buy_tag'] = last_candle['buy_tag']
                            trailing_buy['start_trailing_time'] = datetime.now(timezone.utc)
                            trailing_buy['offset'] = 0

                            self.trailing_buy_info(pair, current_price)
                            log.info(f'start trailing buy for {pair} at {last_candle["close"]}')

                        elif trailing_buy['trailing_buy_order_started']:
                            if trailing_buy_offset == 'forcebuy':
                                # buy in custom conditions
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                log.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

                            elif trailing_buy_offset is None:
                                # stop trailing buy custom conditions
                                self.trailing_buy(pair, reinit=True)
                                log.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')

                            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                # update uplimit
                                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                                log.info(f'update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                            elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                                # buy ! current price > uplimit && lower thant starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                log.info(f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price price ({(trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy))}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")

                            elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                                # stop trailing buy because price is too high
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                                log.info(f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_buy_info(pair, current_price)
                                log.info(f'price too high for {pair} !')

                    else:
                        log.info(f"Wait for next buy signal for {pair}")

                if (val == True):
                    self.trailing_buy_info(pair, rate)
                    self.trailing_buy(pair, reinit=True)
                    log.info(f'STOP trailing buy for {pair} because I buy it')

        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_buy = self.trailing_buy(metadata['pair'])
            if (last_candle['buy'] == 1):
                if not trailing_buy['trailing_buy_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                    if not open_trades:
                        log.info(f"Set 'allow_trailing' to True for {metadata['pair']} to start trailing!!!")
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_buy['allow_trailing'] = True
                        initial_buy_tag = last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal'
                        dataframe.loc[:, 'buy_tag'] = f"{initial_buy_tag} (start trail price {last_candle['close']})"
            else:
                if (trailing_buy['trailing_buy_order_started'] == True):
                    log.info(f"Continue trailing for {metadata['pair']}. Manually trigger buy signal!!")
                    dataframe.loc[:,'buy'] = 1
                    dataframe.loc[:, 'buy_tag'] = trailing_buy['buy_tag']
                    # dataframe['buy'] = 1
                    #idk its the right place here nut yea

        return dataframe

# Elliot Wave Oscillator
def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close'] * 100
    return smadif

# Exponential moving average of a volume weighted simple moving average
def ema_vwma_osc(dataframe, len_slow_ma):
    slow_ema = Series(ta.EMA(vwma(dataframe, len_slow_ma), len_slow_ma))
    return ((slow_ema - slow_ema.shift(1)) / slow_ema.shift(1)) * 100

def pivot_points(dataframe: DataFrame, mode = 'fibonacci') -> Series:
    hlc3_pivot = (dataframe['high'] + dataframe['low'] + dataframe['close']).shift(1) / 3
    hl_range = (dataframe['high'] - dataframe['low']).shift(1)
    if mode == 'simple':
        res1 = hlc3_pivot * 2 - dataframe['low'].shift(1)
        sup1 = hlc3_pivot * 2 - dataframe['high'].shift(1)
        res2 = hlc3_pivot + (dataframe['high'] - dataframe['low']).shift()
        sup2 = hlc3_pivot - (dataframe['high'] - dataframe['low']).shift()
        res3 = hlc3_pivot * 2 + (dataframe['high'] - 2 * dataframe['low']).shift()
        sup3 = hlc3_pivot * 2 - (2 * dataframe['high'] - dataframe['low']).shift()
    elif mode == 'fibonacci':
        res1 = hlc3_pivot + 0.382 * hl_range
        sup1 = hlc3_pivot - 0.382 * hl_range
        res2 = hlc3_pivot + 0.618 * hl_range
        sup2 = hlc3_pivot - 0.618 * hl_range
        res3 = hlc3_pivot + 1 * hl_range
        sup3 = hlc3_pivot - 1 * hl_range

    return hlc3_pivot, res1, res2, res3, sup1, sup2, sup3

def heikin_ashi(dataframe, smooth_inputs = False, smooth_outputs = False, length = 10):
    df = dataframe[['open','close','high','low']].copy().fillna(0)
    if smooth_inputs:
        df['open_s']  = ta.EMA(df['open'], timeframe = length)
        df['high_s']  = ta.EMA(df['high'], timeframe = length)
        df['low_s']   = ta.EMA(df['low'],  timeframe = length)
        df['close_s'] = ta.EMA(df['close'],timeframe = length)

        open_ha  = (df['open_s'].shift(1) + df['close_s'].shift(1)) / 2
        high_ha  = df.loc[:, ['high_s', 'open_s', 'close_s']].max(axis=1)
        low_ha   = df.loc[:, ['low_s', 'open_s', 'close_s']].min(axis=1)
        close_ha = (df['open_s'] + df['high_s'] + df['low_s'] + df['close_s'])/4
    else:
        open_ha  = (df['open'].shift(1) + df['close'].shift(1)) / 2
        high_ha  = df.loc[:, ['high', 'open', 'close']].max(axis=1)
        low_ha   = df.loc[:, ['low', 'open', 'close']].min(axis=1)
        close_ha = (df['open'] + df['high'] + df['low'] + df['close'])/4

    open_ha = open_ha.fillna(0)
    high_ha = high_ha.fillna(0)
    low_ha  = low_ha.fillna(0)
    close_ha = close_ha.fillna(0)

    if smooth_outputs:
        open_sha  = ta.EMA(open_ha, timeframe = length)
        high_sha  = ta.EMA(high_ha, timeframe = length)
        low_sha   = ta.EMA(low_ha, timeframe = length)
        close_sha = ta.EMA(close_ha, timeframe = length)

        return open_sha, close_sha, low_sha
    else:
        return open_ha, close_ha, low_ha

# PMAX
def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx

# Mom DIV
def momdiv(dataframe: DataFrame, mom_length: int = 10, bb_length: int = 20, bb_dev: float = 2.0, lookback: int = 30) -> DataFrame:
    mom: Series = ta.MOM(dataframe, timeperiod=mom_length)
    upperband, middleband, lowerband = ta.BBANDS(mom, timeperiod=bb_length, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0)
    buy = qtpylib.crossed_below(mom, lowerband)
    sell = qtpylib.crossed_above(mom, upperband)
    hh = dataframe['high'].rolling(lookback).max()
    ll = dataframe['low'].rolling(lookback).min()
    coh = dataframe['high'] >= hh
    col = dataframe['low'] <= ll
    df = DataFrame({
            "momdiv_mom": mom,
            "momdiv_upperb": upperband,
            "momdiv_lowerb": lowerband,
            "momdiv_buy": buy,
            "momdiv_sell": sell,
            "momdiv_coh": coh,
            "momdiv_col": col,
        }, index=dataframe['close'].index)
    return df

def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']