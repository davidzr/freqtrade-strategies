import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, CategoricalParameter)
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime


###########################################################################################################
##                NostalgiaForInfinityV4 by iterativ                                                     ##
##                                                                                                       ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##
##                                                                                                       ##
###########################################################################################################
##               GENERAL RECOMMENDATIONS                                                                 ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 4 and 6 open trades, with unlimited stake.        ##
##   A pairlist with 40 to 80 pairs. Volume pairlist works well.                                         ##
##   Prefer stable coin (USDT, BUSDT etc) pairs, instead of BTC or ETH pairs.                            ##
##   Highly recommended to blacklist leveraged tokens (*BULL, *BEAR, *UP, *DOWN etc).                    ##
##   Ensure that you don't override any variables in you config.json. Especially                         ##
##   the timeframe (must be 5m).                                                                         ##
##     use_sell_signal must set to true (or not set at all).                                             ##
##     sell_profit_only must set to false (or not set at all).                                           ##
##     ignore_roi_if_buy_signal must set to true (or not set at all).                                    ##
##                                                                                                       ##
###########################################################################################################
##               DONATIONS                                                                               ##
##                                                                                                       ##
##   Absolutely not required. However, will be accepted as a token of appreciation.                      ##
##                                                                                                       ##
##   BTC: bc1qvflsvddkmxh7eqhc4jyu5z5k6xcw3ay8jl49sk                                                     ##
##   ETH (ERC20): 0x83D3cFb8001BDC5d2211cBeBB8cB3461E5f7Ec91                                             ##
##   BEP20/BSC (ETH, BNB, ...): 0x86A0B21a20b39d16424B7c8003E4A7e12d78ABEe                               ##
##                                                                                                       ##
###########################################################################################################


class NFI46Offset(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.013,
    }

    stoploss = -0.99

    # Multi Offset
    base_nb_candles_buy = IntParameter(
        5, 80, default=20, load=True, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=20, load=True, space='sell', optimize=True)
    low_offset_sma = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_sma = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_ema = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_ema = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_trima = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_trima = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_t3 = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_t3 = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_kama = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_kama = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)

    # Protection
    ewo_low = DecimalParameter(
        -20.0, -8.0, default=-20.0, load=True, space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=6.0, load=True, space='buy', optimize=True)
    fast_ewo = IntParameter(
        10, 50, default=50, load=True, space='buy', optimize=True)
    slow_ewo = IntParameter(
        100, 200, default=200, load=True, space='buy', optimize=True)

    # MA list
    ma_types = ['sma', 'ema', 'trima', 't3', 'kama']
    ma_map = {
        'sma': {
            'low_offset': low_offset_sma.value,
            'high_offset': high_offset_sma.value,
            'calculate': ta.SMA
        },
        'ema': {
            'low_offset': low_offset_ema.value,
            'high_offset': high_offset_ema.value,
            'calculate': ta.EMA
        },
        'trima': {
            'low_offset': low_offset_trima.value,
            'high_offset': high_offset_trima.value,
            'calculate': ta.TRIMA
        },
        't3': {
            'low_offset': low_offset_t3.value,
            'high_offset': high_offset_t3.value,
            'calculate': ta.T3
        },
        'kama': {
            'low_offset': low_offset_kama.value,
            'high_offset': high_offset_kama.value,
            'calculate': ta.KAMA
        }
    }

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    # Custom Stoploss
    use_custom_stoploss = False
    
    # Optimal timeframe for the strategy.
    timeframe = '5m'
    inf_1h = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 400

    # plot config
    plot_config = {
        'main_plot': {
            'ma_offset_buy': {'color': 'orange'},
            'ma_offset_sell': {'color': 'orange'},
        },
    }

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'trailing_stop_loss': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    #############################################################

    buy_params = {
        #############
        # Enable/Disable conditions
        "buy_condition_1_enable": True,
        "buy_condition_2_enable": True,
        "buy_condition_3_enable": True,
        "buy_condition_4_enable": True,
        "buy_condition_5_enable": True,
        "buy_condition_6_enable": True,
        "buy_condition_7_enable": True,
        "buy_condition_8_enable": True,
        "buy_condition_9_enable": True,
        "buy_condition_10_enable": True,
        "buy_condition_11_enable": True,
        "buy_condition_12_enable": True,
        "buy_condition_13_enable": True,
        "buy_condition_14_enable": True,
        "buy_condition_15_enable": True,
        "buy_condition_16_enable": True,
        "buy_condition_17_enable": True,
        # Hyperopt
        # Multi Offset
        "base_nb_candles_buy": 47,
        "buy_chop_min_19": 41.0,
        "buy_rsi_1h_min_19": 50.9,
        "ewo_high": 3.28,
        "ewo_low": -16.004,
        "low_offset_ema": 0.93,
        "low_offset_kama": 0.985,
        "low_offset_sma": 0.9,
        "low_offset_t3": 0.975,
        "low_offset_trima": 0.973,
    }

    sell_params = {
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
        # Hyperopt
        # Multi Offset
        "base_nb_candles_sell": 34,
        "high_offset_ema": 1.047,
        "high_offset_kama": 1.07,
        "high_offset_sma": 1.051,
        "high_offset_t3": 0.999,
        "high_offset_trima": 1.096,
    }

    #############################################################

    buy_condition_1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_5_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_6_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_7_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_8_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_9_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_10_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_11_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_12_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_13_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_14_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_15_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_16_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)
    buy_condition_17_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=True, load=True)

    # Normal dips
    buy_dip_threshold_1 = DecimalParameter(0.001, 0.05, default=0.02, space='buy', decimals=3, optimize=True, load=True)
    buy_dip_threshold_2 = DecimalParameter(0.01, 0.2, default=0.14, space='buy', decimals=3, optimize=True, load=True)
    buy_dip_threshold_3 = DecimalParameter(0.05, 0.4, default=0.32, space='buy', decimals=3, optimize=True, load=True)
    buy_dip_threshold_4 = DecimalParameter(0.2, 0.5, default=0.5, space='buy', decimals=3, optimize=True, load=True)
    # Strict dips
    buy_dip_threshold_5 = DecimalParameter(0.001, 0.05, default=0.015, space='buy', decimals=3, optimize=True, load=True)
    buy_dip_threshold_6 = DecimalParameter(0.01, 0.2, default=0.06, space='buy', decimals=3, optimize=True, load=True)
    buy_dip_threshold_7 = DecimalParameter(0.05, 0.4, default=0.24, space='buy', decimals=3, optimize=True, load=True)
    buy_dip_threshold_8 = DecimalParameter(0.2, 0.5, default=0.4, space='buy', decimals=3, optimize=True, load=True)
    # Loose dips
    buy_dip_threshold_9 = DecimalParameter(0.001, 0.05, default=0.026, space='buy', decimals=3, optimize=True, load=True)
    buy_dip_threshold_10 = DecimalParameter(0.01, 0.2, default=0.24, space='buy', decimals=3, optimize=True, load=True)
    buy_dip_threshold_11 = DecimalParameter(0.05, 0.4, default=0.42, space='buy', decimals=3, optimize=True, load=True)
    buy_dip_threshold_12 = DecimalParameter(0.2, 0.5, default=0.8, space='buy', decimals=3, optimize=True, load=True)

    # 12 hours
    buy_pump_pull_threshold_1 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=True, load=True)
    buy_pump_threshold_1 = DecimalParameter(0.4, 1.0, default=0.46, space='buy', decimals=3, optimize=True, load=True)
    # 36 hours
    buy_pump_pull_threshold_2 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=True, load=True)
    buy_pump_threshold_2 = DecimalParameter(0.4, 1.0, default=0.56, space='buy', decimals=3, optimize=True, load=True)
    # 48 hours
    buy_pump_pull_threshold_3 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=True, load=True)
    buy_pump_threshold_3 = DecimalParameter(0.4, 1.0, default=0.85, space='buy', decimals=3, optimize=True, load=True)

    # 12 hours strict
    buy_pump_pull_threshold_4 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=True, load=True)
    buy_pump_threshold_4 = DecimalParameter(0.4, 1.0, default=0.4, space='buy', decimals=3, optimize=True, load=True)
    # 36 hours strict
    buy_pump_pull_threshold_5 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=True, load=True)
    buy_pump_threshold_5 = DecimalParameter(0.4, 1.0, default=0.56, space='buy', decimals=3, optimize=True, load=True)
    # 48 hours strict
    buy_pump_pull_threshold_6 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=True, load=True)
    buy_pump_threshold_6 = DecimalParameter(0.4, 1.0, default=0.68, space='buy', decimals=3, optimize=True, load=True)
    # 24 hours loose
    buy_pump_pull_threshold_7 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=True, load=True)
    buy_pump_threshold_7 = DecimalParameter(0.4, 1.0, default=0.66, space='buy', decimals=3, optimize=True, load=True)
    # 36 hours loose
    buy_pump_pull_threshold_8 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=True, load=True)
    buy_pump_threshold_8 = DecimalParameter(0.4, 1.0, default=0.7, space='buy', decimals=3, optimize=True, load=True)
    # 48 hours loose
    buy_pump_pull_threshold_9 = DecimalParameter(1.3, 2.0, default=1.4, space='buy', decimals=2, optimize=True, load=True)
    buy_pump_threshold_9 = DecimalParameter(0.4, 1.8, default=1.6, space='buy', decimals=3, optimize=True, load=True)

    buy_min_inc_1 = DecimalParameter(0.01, 0.05, default=0.022, space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_1h_min_1 = DecimalParameter(25.0, 40.0, default=30.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_max_1 = DecimalParameter(70.0, 90.0, default=80.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1 = DecimalParameter(20.0, 40.0, default=36.0, space='buy', decimals=1, optimize=True, load=True)
    buy_mfi_1 = DecimalParameter(20.0, 56.0, default=26.0, space='buy', decimals=1, optimize=True, load=True)

    buy_volume_2 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_min_2 = DecimalParameter(30.0, 40.0, default=36.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_max_2 = DecimalParameter(70.0, 95.0, default=90.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_diff_2 = DecimalParameter(30.0, 50.0, default=34.0, space='buy', decimals=1, optimize=True, load=True)
    buy_mfi_2 = DecimalParameter(30.0, 65.0, default=56.0, space='buy', decimals=1, optimize=True, load=True)
    buy_bb_offset_2 = DecimalParameter(0.97, 0.99, default=0.983, space='buy', decimals=3, optimize=True, load=True)

    buy_bb40_bbdelta_close_3 = DecimalParameter(0.005, 0.06, default=0.057, space='buy', optimize=True, load=True)
    buy_bb40_closedelta_close_3 = DecimalParameter(0.01, 0.03, default=0.023, space='buy', optimize=True, load=True)
    buy_bb40_tail_bbdelta_3 = DecimalParameter(0.15, 0.45, default=0.418, space='buy', optimize=True, load=True)
    buy_ema_rel_3 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=True, load=True)

    buy_bb20_close_bblowerband_4 = DecimalParameter(0.9, 0.99, default=0.979, space='buy', optimize=True, load=True)
    buy_bb20_volume_4 = IntParameter(16, 35, default=18, space='buy', optimize=True, load=True)

    buy_volume_5 = DecimalParameter(1.0, 10.0, default=6.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ema_open_mult_5 = DecimalParameter(0.016, 0.03, default=0.019, space='buy', decimals=3, optimize=True, load=True)
    buy_bb_offset_5 = DecimalParameter(0.98, 1.0, default=0.999, space='buy', decimals=3, optimize=True, load=True)
    buy_ema_rel_5 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=True, load=True)

    buy_volume_6 = DecimalParameter(1.0, 10.0, default=1.5, space='buy', decimals=1, optimize=True, load=True)
    buy_ema_open_mult_6 = DecimalParameter(0.03, 0.04, default=0.025, space='buy', decimals=3, optimize=True, load=True)
    buy_bb_offset_6 = DecimalParameter(0.98, 0.999, default=0.995, space='buy', decimals=3, optimize=True, load=True)

    buy_volume_7 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ema_open_mult_7 = DecimalParameter(0.02, 0.04, default=0.03, space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_7 = DecimalParameter(24.0, 50.0, default=36.0, space='buy', decimals=1, optimize=True, load=True)

    buy_rsi_8 = DecimalParameter(30.0, 50.0, default=46.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ema_rel_8 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=True, load=True)

    buy_volume_9 = DecimalParameter(1.0, 4.0, default=2.0, space='buy', decimals=2, optimize=True, load=True)
    buy_ma_offset_9 = DecimalParameter(0.94, 0.99, default=0.958, space='buy', decimals=3, optimize=True, load=True)
    buy_bb_offset_9 = DecimalParameter(0.97, 0.99, default=0.984, space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_1h_min_9 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_max_9 = DecimalParameter(70.0, 90.0, default=80.0, space='buy', decimals=1, optimize=True, load=True)
    buy_mfi_9 = DecimalParameter(36.0, 65.0, default=56.0, space='buy', decimals=1, optimize=True, load=True)

    buy_volume_10 = DecimalParameter(1.0, 26.0, default=23.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ma_offset_10 = DecimalParameter(0.93, 0.97, default=0.94, space='buy', decimals=3, optimize=True, load=True)
    buy_bb_offset_10 = DecimalParameter(0.97, 0.99, default=0.994, space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_1h_10 = DecimalParameter(20.0, 40.0, default=39.0, space='buy', decimals=1, optimize=True, load=True)

    buy_ma_offset_11 = DecimalParameter(0.93, 0.99, default=0.938, space='buy', decimals=3, optimize=True, load=True)
    buy_min_inc_11 = DecimalParameter(0.005, 0.05, default=0.01, space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_1h_min_11 = DecimalParameter(40.0, 60.0, default=55.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_1h_max_11 = DecimalParameter(70.0, 90.0, default=82.0, space='buy', decimals=1, optimize=True, load=True)
    buy_rsi_11 = DecimalParameter(30.0, 48.0, default=46.0, space='buy', decimals=1, optimize=True, load=True)
    buy_mfi_11 = DecimalParameter(36.0, 56.0, default=38.0, space='buy', decimals=1, optimize=True, load=True)

    buy_volume_12 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ma_offset_12 = DecimalParameter(0.93, 0.97, default=0.936, space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_12 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ewo_12 = DecimalParameter(2.0, 6.0, default=2.8, space='buy', decimals=1, optimize=True, load=True)

    buy_ma_offset_13 = DecimalParameter(0.93, 0.98, default=0.952, space='buy', decimals=3, optimize=True, load=True)
    buy_ewo_13 = DecimalParameter(-14.0, -7.0, default=-7.9, space='buy', decimals=1, optimize=True, load=True)

    buy_volume_14 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ema_open_mult_14 = DecimalParameter(0.01, 0.03, default=0.014, space='buy', decimals=3, optimize=True, load=True)
    buy_bb_offset_14 = DecimalParameter(0.98, 1.0, default=0.992, space='buy', decimals=3, optimize=True, load=True)
    buy_ma_offset_14 = DecimalParameter(0.93, 0.99, default=0.998, space='buy', decimals=3, optimize=True, load=True)

    buy_ema_open_mult_15 = DecimalParameter(0.02, 0.04, default=0.026, space='buy', decimals=3, optimize=True, load=True)
    buy_ma_offset_15 = DecimalParameter(0.93, 0.99, default=0.985, space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_15 = DecimalParameter(30.0, 50.0, default=32.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ema_rel_15 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=True, load=True)

    buy_volume_16 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ma_offset_16 = DecimalParameter(0.93, 0.97, default=0.95, space='buy', decimals=3, optimize=True, load=True)
    buy_rsi_16 = DecimalParameter(26.0, 50.0, default=38.0, space='buy', decimals=1, optimize=True, load=True)
    buy_ewo_16 = DecimalParameter(4.0, 8.0, default=3.6, space='buy', decimals=1, optimize=True, load=True)

    buy_ma_offset_17 = DecimalParameter(0.93, 0.98, default=0.958, space='buy', decimals=3, optimize=True, load=True)
    buy_ewo_17 = DecimalParameter(-18.0, -10.0, default=-12.0, space='buy', decimals=1, optimize=True, load=True)

    # Sell

    sell_condition_1_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_5_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_6_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_7_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell_condition_8_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)

    sell_rsi_bb_1 = DecimalParameter(60.0, 80.0, default=79.5, space='sell', decimals=1, optimize=True, load=True)

    sell_rsi_bb_2 = DecimalParameter(72.0, 90.0, default=81, space='sell', decimals=1, optimize=True, load=True)

    sell_rsi_main_3 = DecimalParameter(77.0, 90.0, default=82, space='sell', decimals=1, optimize=True, load=True)

    sell_dual_rsi_rsi_4 = DecimalParameter(72.0, 84.0, default=73.4, space='sell', decimals=1, optimize=True, load=True)
    sell_dual_rsi_rsi_1h_4 = DecimalParameter(78.0, 92.0, default=79.6, space='sell', decimals=1, optimize=True, load=True)

    sell_ema_relative_5 = DecimalParameter(0.005, 0.05, default=0.024, space='sell', optimize=True, load=True)
    sell_rsi_diff_5 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=True, load=True)

    sell_rsi_under_6 = DecimalParameter(72.0, 90.0, default=79.0, space='sell', decimals=1, optimize=True, load=True)

    sell_rsi_1h_7 = DecimalParameter(80.0, 95.0, default=81.7, space='sell', decimals=1, optimize=True, load=True)

    sell_bb_relative_8 = DecimalParameter(1.05, 1.3, default=1.1, space='sell', decimals=3, optimize=True, load=True)

    sell_custom_profit_0 = DecimalParameter(0.01, 0.1, default=0.01, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_0 = DecimalParameter(30.0, 40.0, default=33.0, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_profit_1 = DecimalParameter(0.01, 0.1, default=0.02, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_1 = DecimalParameter(30.0, 50.0, default=34.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_2 = DecimalParameter(0.01, 0.1, default=0.03, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_2 = DecimalParameter(30.0, 50.0, default=38.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_3 = DecimalParameter(0.01, 0.1, default=0.04, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_3 = DecimalParameter(30.0, 50.0, default=42.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_4 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_4 = DecimalParameter(35.0, 50.0, default=43.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_5 = DecimalParameter(0.01, 0.1, default=0.06, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_5 = DecimalParameter(35.0, 50.0, default=44.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_6 = DecimalParameter(0.01, 0.1, default=0.07, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_6 = DecimalParameter(38.0, 55.0, default=49.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_7 = DecimalParameter(0.01, 0.1, default=0.08, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_7 = DecimalParameter(40.0, 58.0, default=54.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_8 = DecimalParameter(0.06, 0.1, default=0.09, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_8 = DecimalParameter(40.0, 50.0, default=54.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_9 = DecimalParameter(0.05, 0.14, default=0.1, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_9 = DecimalParameter(40.0, 60.0, default=50.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_10 = DecimalParameter(0.1, 0.14, default=0.12, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_10 = DecimalParameter(38.0, 50.0, default=42.0, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_profit_11 = DecimalParameter(0.16, 0.45, default=0.20, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_rsi_11 = DecimalParameter(28.0, 40.0, default=34.0, space='sell', decimals=2, optimize=True, load=True)

    # Profit under EMA200
    sell_custom_under_profit_0 = DecimalParameter(0.01, 0.4, default=0.01, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_0 = DecimalParameter(28.0, 40.0, default=33.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_1 = DecimalParameter(0.01, 0.10, default=0.02, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_1 = DecimalParameter(36.0, 60.0, default=56.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_2 = DecimalParameter(0.01, 0.10, default=0.03, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_2 = DecimalParameter(46.0, 66.0, default=57.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_3 = DecimalParameter(0.01, 0.10, default=0.04, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_3 = DecimalParameter(50.0, 68.0, default=58.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_4 = DecimalParameter(0.02, 0.1, default=0.05, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_4 = DecimalParameter(50.0, 68.0, default=59.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_5 = DecimalParameter(0.02, 0.1, default=0.06, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_5 = DecimalParameter(46.0, 62.0, default=58.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_6 = DecimalParameter(0.03, 0.1, default=0.07, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_6 = DecimalParameter(44.0, 60.0, default=56.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_7 = DecimalParameter(0.04, 0.1, default=0.08, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_7 = DecimalParameter(46.0, 60.0, default=54.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_8 = DecimalParameter(0.06, 0.12, default=0.09, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_8 = DecimalParameter(40.0, 58.0, default=50.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_9 = DecimalParameter(0.08, 0.14, default=0.1, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_9 = DecimalParameter(32.0, 48.0, default=44.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_10 = DecimalParameter(0.1, 0.16, default=0.12, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_10 = DecimalParameter(30.0, 50.0, default=42.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_under_profit_11 = DecimalParameter(0.16, 0.3, default=0.2, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_under_rsi_11 = DecimalParameter(24.0, 40.0, default=34.0, space='sell', decimals=1, optimize=True, load=True)

    # Profit targets for pumped pairs 48h 1
    sell_custom_pump_profit_1_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_1_1 = DecimalParameter(26.0, 40.0, default=34.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_1_2 = DecimalParameter(0.01, 0.6, default=0.02, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_1_2 = DecimalParameter(36.0, 50.0, default=40.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_1_3 = DecimalParameter(0.02, 0.10, default=0.04, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_1_3 = DecimalParameter(38.0, 50.0, default=42.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_1_4 = DecimalParameter(0.06, 0.12, default=0.1, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_1_4 = DecimalParameter(36.0, 48.0, default=42.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_1_5 = DecimalParameter(0.14, 0.24, default=0.2, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_1_5 = DecimalParameter(20.0, 40.0, default=34.0, space='sell', decimals=1, optimize=True, load=True)

    # Profit targets for pumped pairs 36h 1
    sell_custom_pump_profit_2_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_2_1 = DecimalParameter(26.0, 40.0, default=34.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_2_2 = DecimalParameter(0.01, 0.6, default=0.02, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_2_2 = DecimalParameter(36.0, 50.0, default=40.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_2_3 = DecimalParameter(0.02, 0.10, default=0.04, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_2_3 = DecimalParameter(38.0, 50.0, default=40.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_2_4 = DecimalParameter(0.06, 0.12, default=0.1, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_2_4 = DecimalParameter(36.0, 48.0, default=42.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_2_5 = DecimalParameter(0.14, 0.24, default=0.2, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_2_5 = DecimalParameter(20.0, 40.0, default=34.0, space='sell', decimals=1, optimize=True, load=True)

    # Profit targets for pumped pairs 24h 1
    sell_custom_pump_profit_3_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_3_1 = DecimalParameter(26.0, 40.0, default=34.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_3_2 = DecimalParameter(0.01, 0.6, default=0.02, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_3_2 = DecimalParameter(34.0, 50.0, default=40.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_3_3 = DecimalParameter(0.02, 0.10, default=0.04, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_3_3 = DecimalParameter(38.0, 50.0, default=40.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_3_4 = DecimalParameter(0.06, 0.12, default=0.1, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_3_4 = DecimalParameter(36.0, 48.0, default=42.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_profit_3_5 = DecimalParameter(0.14, 0.24, default=0.2, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_rsi_3_5 = DecimalParameter(20.0, 40.0, default=34.0, space='sell', decimals=1, optimize=True, load=True)

    # SMA descending
    sell_custom_dec_profit_min_1 = DecimalParameter(0.01, 0.10, default=0.05, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_dec_profit_max_1 = DecimalParameter(0.06, 0.16, default=0.12, space='sell', decimals=3, optimize=True, load=True)

    # Under EMA100
    sell_custom_dec_profit_min_2 = DecimalParameter(0.05, 0.12, default=0.07, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_dec_profit_max_2 = DecimalParameter(0.06, 0.2, default=0.16, space='sell', decimals=3, optimize=True, load=True)

    # Trail 1
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.2, default=0.16, space='sell', decimals=2, optimize=True, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.4, 0.7, default=0.6, space='sell', decimals=2, optimize=True, load=True)
    sell_trail_down_1 = DecimalParameter(0.01, 0.08, default=0.03, space='sell', decimals=3, optimize=True, load=True)
    sell_trail_rsi_min_1 = DecimalParameter(16.0, 36.0, default=20.0, space='sell', decimals=1, optimize=True, load=True)
    sell_trail_rsi_max_1 = DecimalParameter(30.0, 50.0, default=50.0, space='sell', decimals=1, optimize=True, load=True)

    # Trail 2
    sell_trail_profit_min_2 = DecimalParameter(0.08, 0.16, default=0.1, space='sell', decimals=3, optimize=True, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.3, 0.5, default=0.4, space='sell', decimals=2, optimize=True, load=True)
    sell_trail_down_2 = DecimalParameter(0.02, 0.08, default=0.03, space='sell', decimals=3, optimize=True, load=True)
    sell_trail_rsi_min_2 = DecimalParameter(16.0, 36.0, default=20.0, space='sell', decimals=1, optimize=True, load=True)
    sell_trail_rsi_max_2 = DecimalParameter(30.0, 50.0, default=50.0, space='sell', decimals=1, optimize=True, load=True)

    # Trail 3
    sell_trail_profit_min_3 = DecimalParameter(0.01, 0.12, default=0.06, space='sell', decimals=3, optimize=True, load=True)
    sell_trail_profit_max_3 = DecimalParameter(0.1, 0.3, default=0.2, space='sell', decimals=2, optimize=True, load=True)
    sell_trail_down_3 = DecimalParameter(0.01, 0.06, default=0.05, space='sell', decimals=3, optimize=True, load=True)

    # Under & near EMA200, accept profit
    sell_custom_profit_under_rel_1 = DecimalParameter(0.01, 0.04, default=0.024, space='sell', optimize=True, load=True)
    sell_custom_profit_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=True, load=True)

    # Under & near EMA200, take the loss
    sell_custom_stoploss_under_rel_1 = DecimalParameter(0.001, 0.02, default=0.004, space='sell', optimize=True, load=True)
    sell_custom_stoploss_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=8.0, space='sell', optimize=True, load=True)

    # 48h for pump sell checks
    sell_pump_threshold_1 = DecimalParameter(0.5, 1.2, default=0.9, space='sell', decimals=2, optimize=True, load=True)
    sell_pump_threshold_2 = DecimalParameter(0.4, 0.9, default=0.7, space='sell', decimals=2, optimize=True, load=True)
    sell_pump_threshold_3 = DecimalParameter(0.3, 0.7, default=0.5, space='sell', decimals=2, optimize=True, load=True)

    # 36h for pump sell checks
    sell_pump_threshold_4 = DecimalParameter(0.5, 0.9, default=0.72, space='sell', decimals=2, optimize=True, load=True)
    sell_pump_threshold_5 = DecimalParameter(3.0, 6.0, default=4.0, space='sell', decimals=2, optimize=True, load=True)
    sell_pump_threshold_6 = DecimalParameter(0.8, 1.6, default=1.0, space='sell', decimals=2, optimize=True, load=True)

    # 24h for pump sell checks
    sell_pump_threshold_7 = DecimalParameter(0.5, 0.9, default=0.68, space='sell', decimals=2, optimize=True, load=True)
    sell_pump_threshold_8 = DecimalParameter(0.3, 0.6, default=0.62, space='sell', decimals=2, optimize=True, load=True)
    sell_pump_threshold_9 = DecimalParameter(0.2, 0.5, default=0.3, space='sell', decimals=2, optimize=True, load=True)

    # Pumped, descending SMA
    sell_custom_pump_dec_profit_min_1 = DecimalParameter(0.001, 0.04, default=0.005, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_dec_profit_max_1 = DecimalParameter(0.03, 0.08, default=0.05, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_dec_profit_min_2 = DecimalParameter(0.01, 0.08, default=0.04, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_dec_profit_max_2 = DecimalParameter(0.04, 0.1, default=0.06, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_dec_profit_min_3 = DecimalParameter(0.02, 0.1, default=0.06, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_dec_profit_max_3 = DecimalParameter(0.06, 0.12, default=0.09, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_dec_profit_min_4 = DecimalParameter(0.01, 0.05, default=0.02, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_dec_profit_max_4 = DecimalParameter(0.02, 0.1, default=0.04, space='sell', decimals=3, optimize=True, load=True)


    # Pumped 48h 1, under EMA200
    sell_custom_pump_under_profit_min_1 = DecimalParameter(0.02, 0.06, default=0.04, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_under_profit_max_1 = DecimalParameter(0.04, 0.1, default=0.09, space='sell', decimals=3, optimize=True, load=True)

    # Pumped trail 1
    sell_custom_pump_trail_profit_min_1 = DecimalParameter(0.01, 0.12, default=0.05, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_trail_profit_max_1 = DecimalParameter(0.06, 0.16, default=0.07, space='sell', decimals=2, optimize=True, load=True)
    sell_custom_pump_trail_down_1 = DecimalParameter(0.01, 0.06, default=0.05, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_pump_trail_rsi_min_1 = DecimalParameter(16.0, 36.0, default=20.0, space='sell', decimals=1, optimize=True, load=True)
    sell_custom_pump_trail_rsi_max_1 = DecimalParameter(30.0, 50.0, default=70.0, space='sell', decimals=1, optimize=True, load=True)

    # Stoploss, pumped, 48h 1
    sell_custom_stoploss_pump_max_profit_1 = DecimalParameter(0.01, 0.04, default=0.025, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_stoploss_pump_min_1 = DecimalParameter(-0.1, -0.01, default=-0.02, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_stoploss_pump_max_1 = DecimalParameter(-0.1, -0.01, default=-0.01, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_stoploss_pump_ma_offset_1 = DecimalParameter(0.7, 0.99, default=0.94, space='sell', decimals=2, optimize=True, load=True)

    # Stoploss, pumped, 48h 1
    sell_custom_stoploss_pump_max_profit_2 = DecimalParameter(0.01, 0.04, default=0.025, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_stoploss_pump_loss_2 = DecimalParameter(-0.1, -0.01, default=-0.05, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_stoploss_pump_ma_offset_2 = DecimalParameter(0.7, 0.99, default=0.92, space='sell', decimals=2, optimize=True, load=True)

    # Stoploss, pumped, 36h 3
    sell_custom_stoploss_pump_max_profit_3 = DecimalParameter(0.01, 0.04, default=0.008, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_stoploss_pump_loss_3 = DecimalParameter(-0.16, -0.06, default=-0.12, space='sell', decimals=3, optimize=True, load=True)
    sell_custom_stoploss_pump_ma_offset_3 = DecimalParameter(0.7, 0.99, default=0.88, space='sell', decimals=2, optimize=True, load=True)

    #############################################################

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)

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

            elif (self.sell_custom_dec_profit_max_1.value > current_profit > self.sell_custom_dec_profit_min_1.value) & (last_candle['sma_200_dec']):
                return 'signal_profit_d_1'
            elif (self.sell_custom_dec_profit_max_2.value > current_profit > self.sell_custom_dec_profit_min_2.value) & (last_candle['close'] < last_candle['ema_100']):
                return 'signal_profit_d_2'

            # Trailing
            elif (self.sell_trail_profit_max_1.value > current_profit > self.sell_trail_profit_min_1.value) & (self.sell_trail_rsi_min_1.value < last_candle['rsi'] < self.sell_trail_rsi_max_1.value) & (max_profit > (current_profit + self.sell_trail_down_1.value)):
                return 'signal_profit_t_1'
            elif (self.sell_trail_profit_max_2.value > current_profit > self.sell_trail_profit_min_2.value) & (self.sell_trail_rsi_min_2.value < last_candle['rsi'] < self.sell_trail_rsi_max_2.value) & (max_profit > (current_profit + self.sell_trail_down_2.value)):
                return 'signal_profit_t_2'
            elif (self.sell_trail_profit_max_3.value > current_profit > self.sell_trail_profit_min_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)) & (last_candle['sma_200_dec_1h']):
                return 'signal_profit_t_3'

            elif (last_candle['close'] < last_candle['ema_200']) & (current_profit > self.sell_trail_profit_min_3.value) & (current_profit < self.sell_trail_profit_max_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)):
                return 'signal_profit_u_t_1'

            elif (current_profit > 0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_profit_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_profit_under_rsi_diff_1.value):
                return 'signal_profit_u_e_1'

            elif (current_profit < -0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_stoploss_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_under_rsi_diff_1.value):
                return 'signal_stoploss_u_1'

            elif (self.sell_custom_pump_dec_profit_max_1.value > current_profit > self.sell_custom_pump_dec_profit_min_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['sma_200_dec']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_d_1'
            elif (self.sell_custom_pump_dec_profit_max_2.value > current_profit > self.sell_custom_pump_dec_profit_min_2.value) & (last_candle['sell_pump_48_2_1h']) & (last_candle['sma_200_dec']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_d_2'
            elif (self.sell_custom_pump_dec_profit_max_3.value > current_profit > self.sell_custom_pump_dec_profit_min_3.value) & (last_candle['sell_pump_48_3_1h']) & (last_candle['sma_200_dec']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_d_3'
            elif (self.sell_custom_pump_dec_profit_max_4.value > current_profit > self.sell_custom_pump_dec_profit_min_4.value) & (last_candle['sma_200_dec']) & (last_candle['sell_pump_24_2_1h']):
                return 'signal_profit_p_d_4'

            # Pumped 48h 1, under EMA200
            elif (self.sell_custom_pump_under_profit_max_1.value > current_profit > self.sell_custom_pump_under_profit_min_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_u_1'

            # Pumped 36h 2, trail 1
            elif (last_candle['sell_pump_36_2_1h']) & (self.sell_custom_pump_trail_profit_max_1.value > current_profit > self.sell_custom_pump_trail_profit_min_1.value) & (self.sell_custom_pump_trail_rsi_min_1.value < last_candle['rsi'] < self.sell_custom_pump_trail_rsi_max_1.value) & (max_profit > (current_profit + self.sell_custom_pump_trail_down_1.value)):
                return 'signal_profit_p_t_1'

            elif (max_profit < self.sell_custom_stoploss_pump_max_profit_1.value) & (self.sell_custom_stoploss_pump_min_1.value < current_profit < self.sell_custom_stoploss_pump_max_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['sma_200_dec']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_1.value)):
                return 'signal_stoploss_p_1'

            elif (max_profit < self.sell_custom_stoploss_pump_max_profit_2.value) & (current_profit < self.sell_custom_stoploss_pump_loss_2.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['sma_200_dec_1h']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_2.value)):
                return 'signal_stoploss_p_2'

            elif (max_profit < self.sell_custom_stoploss_pump_max_profit_3.value) & (current_profit < self.sell_custom_stoploss_pump_loss_3.value) & (last_candle['sell_pump_36_3_1h']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_3.value)):
                return 'signal_stoploss_p_3'

        return None

    def range_percent_change(self, dataframe: DataFrame, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        df = dataframe.copy()
        return ((df['open'].rolling(length).max() - df['close'].rolling(length).min()) / df['close'].rolling(length).min())

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
        return (self.range_percent_change(df, length) < thresh) | (self.range_maxgap_adjusted(df, length, pull_thresh) > self.range_height(df, length))

    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        # EMA
        informative_1h['ema_12'] = ta.EMA(informative_1h, timeperiod=12)
        informative_1h['ema_15'] = ta.EMA(informative_1h, timeperiod=15)
        informative_1h['ema_20'] = ta.EMA(informative_1h, timeperiod=20)
        informative_1h['ema_26'] = ta.EMA(informative_1h, timeperiod=26)
        informative_1h['ema_35'] = ta.EMA(informative_1h, timeperiod=35)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        # SMA
        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)
        informative_1h['sma_200_dec'] = informative_1h['sma_200'] < informative_1h['sma_200'].shift(20)

        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband'] = bollinger['lower']
        informative_1h['bb_middleband'] = bollinger['mid']
        informative_1h['bb_upperband'] = bollinger['upper']

        # Chaikin Money Flow
        informative_1h['cmf'] = chaikin_money_flow(informative_1h, 20)

        # Pump protections
        informative_1h['safe_pump_24_normal'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_1.value, self.buy_pump_pull_threshold_1.value)
        informative_1h['safe_pump_36_normal'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_2.value, self.buy_pump_pull_threshold_2.value)
        informative_1h['safe_pump_48_normal'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_3.value, self.buy_pump_pull_threshold_3.value)

        informative_1h['safe_pump_24_strict'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_4.value, self.buy_pump_pull_threshold_4.value)
        informative_1h['safe_pump_36_strict'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_5.value, self.buy_pump_pull_threshold_5.value)
        informative_1h['safe_pump_48_strict'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_6.value, self.buy_pump_pull_threshold_6.value)

        informative_1h['safe_pump_24_loose'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_7.value, self.buy_pump_pull_threshold_7.value)
        informative_1h['safe_pump_36_loose'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_8.value, self.buy_pump_pull_threshold_8.value)
        informative_1h['safe_pump_48_loose'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_9.value, self.buy_pump_pull_threshold_9.value)

        informative_1h['safe_pump_24'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_1.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / self.buy_pump_pull_threshold_1.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))
        informative_1h['safe_pump_36'] = ((((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / informative_1h['close'].rolling(36).min()) < self.buy_pump_threshold_2.value) | (((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / self.buy_pump_pull_threshold_2.value) > (informative_1h['close'] - informative_1h['close'].rolling(36).min())))
        informative_1h['safe_pump_48'] = ((((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / informative_1h['close'].rolling(48).min()) < self.buy_pump_threshold_3.value) | (((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / self.buy_pump_pull_threshold_3.value) > (informative_1h['close'] - informative_1h['close'].rolling(48).min())))

        informative_1h['sell_pump_48_1'] = (((informative_1h['high'].rolling(48).max() - informative_1h['low'].rolling(48).min()) / informative_1h['low'].rolling(48).min()) > self.sell_pump_threshold_1.value)
        informative_1h['sell_pump_48_2'] = (((informative_1h['high'].rolling(48).max() - informative_1h['low'].rolling(48).min()) / informative_1h['low'].rolling(48).min()) > self.sell_pump_threshold_2.value)
        informative_1h['sell_pump_48_3'] = (((informative_1h['high'].rolling(48).max() - informative_1h['low'].rolling(48).min()) / informative_1h['low'].rolling(48).min()) > self.sell_pump_threshold_3.value)

        informative_1h['sell_pump_36_1'] = (((informative_1h['high'].rolling(36).max() - informative_1h['low'].rolling(36).min()) / informative_1h['low'].rolling(36).min()) > self.sell_pump_threshold_4.value)
        informative_1h['sell_pump_36_2'] = (((informative_1h['high'].rolling(36).max() - informative_1h['low'].rolling(36).min()) / informative_1h['low'].rolling(36).min()) > self.sell_pump_threshold_5.value)
        informative_1h['sell_pump_36_3'] = (((informative_1h['high'].rolling(36).max() - informative_1h['low'].rolling(36).min()) / informative_1h['low'].rolling(36).min()) > self.sell_pump_threshold_6.value)

        informative_1h['sell_pump_24_1'] = (((informative_1h['high'].rolling(24).max() - informative_1h['low'].rolling(24).min()) / informative_1h['low'].rolling(24).min()) > self.sell_pump_threshold_7.value)
        informative_1h['sell_pump_24_2'] = (((informative_1h['high'].rolling(24).max() - informative_1h['low'].rolling(24).min()) / informative_1h['low'].rolling(24).min()) > self.sell_pump_threshold_8.value)
        informative_1h['sell_pump_24_3'] = (((informative_1h['high'].rolling(24).max() - informative_1h['low'].rolling(24).min()) / informative_1h['low'].rolling(24).min()) > self.sell_pump_threshold_9.value)

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # BB 40
        bb_40 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['lower'] = bb_40['lower']
        dataframe['mid'] = bb_40['mid']
        dataframe['bbdelta'] = (bb_40['mid'] - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        # BB 20
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # EMA 200
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['sma_200_dec'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # EWO
        dataframe['ewo'] = EWO(dataframe, 50, 200)

        # Alligator
        dataframe['lips'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['smma_lips'] = dataframe['lips'].rolling(3).mean()
        dataframe['teeth'] = ta.SMA(dataframe, timeperiod=8)
        dataframe['smma_teeth'] = dataframe['teeth'].rolling(5).mean()
        dataframe['jaw'] = ta.SMA(dataframe, timeperiod=13)
        dataframe['smma_jaw'] = dataframe['jaw'].rolling(8).mean()

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Chopiness
        dataframe['chop']= qtpylib.chopiness(dataframe, 14)

        # Dip protection
        dataframe['safe_dips'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
                                  (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_2.value) &
                                  (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_3.value) &
                                  (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_4.value))


        dataframe['safe_dips_normal'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
                                  (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_2.value) &
                                  (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_3.value) &
                                  (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_4.value))

        dataframe['safe_dips_strict'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_5.value) &
                                  (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_6.value) &
                                  (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_7.value) &
                                  (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_8.value))

        dataframe['safe_dips_loose'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_9.value) &
                                  (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_10.value) &
                                  (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_11.value) &
                                  (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_12.value))

        # Volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)
        dataframe['volume_mean_30'] = dataframe['volume'].rolling(30).mean()

        # Offset
        for i in self.ma_types:
            dataframe[f'{i}_offset_buy'] = self.ma_map[f'{i}']['calculate'](
                dataframe, self.base_nb_candles_buy.value) * \
                self.ma_map[f'{i}']['low_offset']
            dataframe[f'{i}_offset_sell'] = self.ma_map[f'{i}']['calculate'](
                dataframe, self.base_nb_candles_sell.value) * \
                self.ma_map[f'{i}']['high_offset']

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                self.buy_condition_1_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (dataframe['sma_200'] > dataframe['sma_200'].shift(20)) &

                (dataframe['safe_dips']) &
                (dataframe['safe_pump_48_1h']) &

                (((dataframe['close'] - dataframe['open'].rolling(36).min()) / dataframe['open'].rolling(36).min()) > self.buy_min_inc_1.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_1.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_1.value) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_2_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_pump_24_strict_1h']) &

                (dataframe['volume_mean_4'] * self.buy_volume_2.value > dataframe['volume']) &

                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_2.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_2.value) &
                (dataframe['rsi'] < dataframe['rsi_1h'] - self.buy_rsi_1h_diff_2.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_bb_offset_2.value)) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_3_enable.value &

                (dataframe['close'] > (dataframe['ema_200_1h'] * self.buy_ema_rel_3.value)) &
                (dataframe['ema_100'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['ema_100_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_pump_36_1h']) &

                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * self.buy_bb40_bbdelta_close_3.value) &
                dataframe['closedelta'].gt(dataframe['close'] * self.buy_bb40_closedelta_close_3.value) &
                dataframe['tail'].lt(dataframe['bbdelta'] * self.buy_bb40_tail_bbdelta_3.value) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift()) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_4_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_24_1h']) &

                (dataframe['close'] < dataframe['ema_50']) &
                (dataframe['close'] < self.buy_bb20_close_bblowerband_4.value * dataframe['bb_lowerband']) &
                (dataframe['volume'] < (dataframe['volume_mean_30'].shift(1) * self.buy_bb20_volume_4.value))
            )
        )

        conditions.append(
            (
                self.buy_condition_5_enable.value &

                (dataframe['ema_100'] > dataframe['ema_200']) &
                (dataframe['close'] > (dataframe['ema_200_1h'] * self.buy_ema_rel_5.value)) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips']) &
                (dataframe['safe_pump_36_strict_1h']) &

                (dataframe['volume_mean_4'] * self.buy_volume_5.value > dataframe['volume']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_5.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_bb_offset_5.value)) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_6_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips_strict']) &

                (dataframe['volume'].rolling(4).mean() * self.buy_volume_6.value > dataframe['volume']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_6.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_bb_offset_6.value)) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_7_enable.value &

                (dataframe['ema_100'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips']) &

                (dataframe['volume'].rolling(4).mean() * self.buy_volume_6.value > dataframe['volume']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_7.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['rsi'] < self.buy_rsi_7.value) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_8_enable.value &

                (dataframe['close'] > (dataframe['ema_200_1h'] * self.buy_ema_rel_8.value)) &
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(24)) &

                (dataframe['close'] > dataframe['open']) &

                (dataframe['close'] > dataframe['smma_lips']) &

                (dataframe['smma_lips'] > dataframe['smma_teeth']) &
                (dataframe['smma_teeth'] > dataframe['smma_jaw']) &
                (dataframe['smma_lips'].shift(1) > dataframe['smma_teeth'].shift(1)) &
                (dataframe['smma_teeth'].shift(1) > dataframe['smma_jaw'].shift(1)) &

                (dataframe['smma_lips'] > dataframe['smma_lips'].shift(1)) &
                (dataframe['smma_teeth'] > dataframe['smma_teeth'].shift(1)) &
                (dataframe['smma_jaw'] > dataframe['smma_jaw'].shift(1)) &

                (dataframe['rsi'] < self.buy_rsi_8.value) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_9_enable.value &

                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &

                (dataframe['safe_dips_strict']) &

                (dataframe['volume_mean_4'] * self.buy_volume_9.value > dataframe['volume']) &

                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_9.value) &
                (dataframe['close'] < dataframe['bb_lowerband'] * self.buy_bb_offset_9.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_9.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_9.value) &
                (dataframe['mfi'] < self.buy_mfi_9.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_10_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(24)) &

                (dataframe['safe_dips']) &
                (dataframe['safe_pump_24_1h']) &

                ((dataframe['volume_mean_4'] * self.buy_volume_10.value) > dataframe['volume']) &

                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_10.value) &
                (dataframe['close'] < dataframe['bb_lowerband'] * self.buy_bb_offset_10.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_10.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_11_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &

                (dataframe['safe_pump_24_1h']) &

                (((dataframe['close'] - dataframe['open'].rolling(36).min()) / dataframe['open'].rolling(36).min()) > self.buy_min_inc_11.value) &
                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_11.value) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_11.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_max_11.value) &
                (dataframe['rsi'] < self.buy_rsi_11.value) &
                (dataframe['mfi'] < self.buy_mfi_11.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_12_enable.value &

                (dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(24)) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_24_strict_1h']) &

                ((dataframe['volume_mean_4'] * self.buy_volume_12.value) > dataframe['volume']) &

                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_12.value) &
                (dataframe['ewo'] > self.buy_ewo_12.value) &
                (dataframe['rsi'] < self.buy_rsi_12.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_13_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(24)) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_24_strict_1h']) &

                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_13.value) &
                (dataframe['ewo'] < self.buy_ewo_13.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_14_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (dataframe['sma_200'] > dataframe['sma_200'].shift(20)) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_48_1h']) &

                (dataframe['volume_mean_4'] * self.buy_volume_14.value > dataframe['volume']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_14.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_bb_offset_14.value)) &
                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_14.value) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_15_enable.value &

                (dataframe['close'] > dataframe['ema_200_1h'] * self.buy_ema_rel_15.value) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips_strict']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_15.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['rsi'] < self.buy_rsi_15.value) &
                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_15.value) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_16_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_24_strict_1h']) &

                ((dataframe['volume_mean_4'] * self.buy_volume_16.value) > dataframe['volume']) &

                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_16.value) &
                (dataframe['ewo'] > self.buy_ewo_16.value) &
                (dataframe['rsi'] < self.buy_rsi_16.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_17_enable.value &

                (dataframe['safe_dips_strict']) &

                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_17.value) &
                (dataframe['ewo'] < self.buy_ewo_17.value) &
                (dataframe['volume'] > 0)
            )
        )

        for i in self.ma_types:
            conditions.append(
                (
                    dataframe['close'] < dataframe[f'{i}_offset_buy']) &
                (
                    (dataframe['ewo'] < self.ewo_low.value) |
                    (dataframe['ewo'] > self.ewo_high.value)
                ) &
                (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                self.sell_condition_1_enable.value &

                (dataframe['rsi'] > self.sell_rsi_bb_1.value) &
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb_upperband'].shift(2)) &
                (dataframe['close'].shift(3) > dataframe['bb_upperband'].shift(3)) &
                (dataframe['close'].shift(4) > dataframe['bb_upperband'].shift(4)) &
                (dataframe['close'].shift(5) > dataframe['bb_upperband'].shift(5)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_2_enable.value &

                (dataframe['rsi'] > self.sell_rsi_bb_2.value) &
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['close'].shift(1) > dataframe['bb_upperband'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb_upperband'].shift(2)) &
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

                (dataframe['close'] > dataframe['bb_upperband_1h'] * self.sell_bb_relative_8.value) &

                (dataframe['volume'] > 0)
            )
        )

        for i in self.ma_types:
            conditions.append(
                (
                    (dataframe['close'] > dataframe[f'{i}_offset_sell']) &
                    (dataframe['volume'] > 0)
                )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        return dataframe

# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False):
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')