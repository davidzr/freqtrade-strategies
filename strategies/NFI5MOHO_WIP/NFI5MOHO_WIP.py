import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, CategoricalParameter)
from pandas import DataFrame
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime


###########################################################################################################
##                NostalgiaForInfinityV5 by iterativ                                                     ##
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

# 20210624
# NostalgiaForInfinityV5 + MultiOffsetLamboV0 + Hyper-optimized some parameters.

# I hope you do enough testing before proceeding.
# Thank you to those who created these strategies.

class NFI5MOHO_WIP(IStrategy):
    INTERFACE_VERSION = 2

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'trailing_stop_loss': 'limit',
        'stoploss': 'limit',
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
        "buy_condition_18_enable": True,
        "buy_condition_19_enable": True,
        "buy_condition_20_enable": True,
        "buy_condition_21_enable": True,
        # Hyperopt
        # Multi Offset
	"""
	"base_nb_candles_buy": 42,
        "buy_chop_min_19": 29.3,
        "buy_rsi_1h_min_19": 52.4,
        "ewo_high": 5.262,
        "ewo_low": -8.164,
        "low_offset_ema": 0.984,
        "low_offset_kama": 0.919,
        "low_offset_sma": 0.97,
        "low_offset_t3": 0.904,
        "low_offset_trima": 0.984,
	"""
	"base_nb_candles_buy": 72,
        "buy_chop_min_19": 58.2,
        "buy_rsi_1h_min_19": 65.3,
        "ewo_high": 3.319,
        "ewo_low": -11.101,
        "low_offset_ema": 0.929,
        "low_offset_kama": 0.972,
        "low_offset_sma": 0.955,
        "low_offset_t3": 0.975,
        "low_offset_trima": 0.949,
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

    # ROI table:
    minimal_roi = {
        "0": 0.111,
        "13": 0.048,
        "50": 0.015,
        "61": 0.01
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
        10, 50, default=50, load=True, space='buy', optimize=False)
    slow_ewo = IntParameter(
        100, 200, default=200, load=True, space='buy', optimize=False)

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
    startup_candle_count: int = 300

    # plot config
    plot_config = {
        'main_plot': {
            'ma_offset_buy': {'color': 'orange'},
            'ma_offset_sell': {'color': 'orange'},
        },
    }

    #############################################################

    buy_condition_1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_5_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_6_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_7_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_8_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_9_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_10_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_11_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_12_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_13_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_14_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_15_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_16_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_17_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_18_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_19_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_20_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_21_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)

    # Normal dips
    buy_dip_threshold_1 = DecimalParameter(0.001, 0.05, default=0.02, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_2 = DecimalParameter(0.01, 0.2, default=0.14, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_3 = DecimalParameter(0.05, 0.4, default=0.32, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_4 = DecimalParameter(0.2, 0.5, default=0.5, space='buy', decimals=3, optimize=False, load=True)
    # Strict dips
    buy_dip_threshold_5 = DecimalParameter(0.001, 0.05, default=0.015, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_6 = DecimalParameter(0.01, 0.2, default=0.06, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_7 = DecimalParameter(0.05, 0.4, default=0.24, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_8 = DecimalParameter(0.2, 0.5, default=0.4, space='buy', decimals=3, optimize=False, load=True)
    # Loose dips
    buy_dip_threshold_9 = DecimalParameter(0.001, 0.05, default=0.026, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_10 = DecimalParameter(0.01, 0.2, default=0.24, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_11 = DecimalParameter(0.05, 0.4, default=0.42, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_12 = DecimalParameter(0.2, 0.5, default=0.66, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours
    buy_pump_pull_threshold_1 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_1 = DecimalParameter(0.4, 1.0, default=0.5, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours
    buy_pump_pull_threshold_2 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_2 = DecimalParameter(0.4, 1.0, default=0.56, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours
    buy_pump_pull_threshold_3 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_3 = DecimalParameter(0.4, 1.0, default=0.85, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours strict
    buy_pump_pull_threshold_4 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_4 = DecimalParameter(0.4, 1.0, default=0.4, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours strict
    buy_pump_pull_threshold_5 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_5 = DecimalParameter(0.4, 1.0, default=0.56, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours strict
    buy_pump_pull_threshold_6 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_6 = DecimalParameter(0.4, 1.0, default=0.68, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours loose
    buy_pump_pull_threshold_7 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_7 = DecimalParameter(0.4, 1.0, default=0.66, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours loose
    buy_pump_pull_threshold_8 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_8 = DecimalParameter(0.4, 1.0, default=0.7, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours loose
    buy_pump_pull_threshold_9 = DecimalParameter(1.5, 3.0, default=1.4, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_9 = DecimalParameter(0.4, 1.8, default=1.3, space='buy', decimals=3, optimize=False, load=True)

    buy_min_inc_1 = DecimalParameter(0.01, 0.05, default=0.022, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_1 = DecimalParameter(25.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_1 = DecimalParameter(70.0, 90.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1 = DecimalParameter(20.0, 40.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_1 = DecimalParameter(20.0, 40.0, default=26.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_2 = DecimalParameter(1.0, 10.0, default=2.6, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_min_2 = DecimalParameter(30.0, 40.0, default=32.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_2 = DecimalParameter(70.0, 95.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_diff_2 = DecimalParameter(30.0, 50.0, default=39.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_2 = DecimalParameter(30.0, 56.0, default=49.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_2 = DecimalParameter(0.97, 0.999, default=0.983, space='buy', decimals=3, optimize=False, load=True)

    buy_bb40_bbdelta_close_3 = DecimalParameter(0.005, 0.06, default=0.057, space='buy', optimize=False, load=True)
    buy_bb40_closedelta_close_3 = DecimalParameter(0.01, 0.03, default=0.023, space='buy', optimize=False, load=True)
    buy_bb40_tail_bbdelta_3 = DecimalParameter(0.15, 0.45, default=0.418, space='buy', optimize=False, load=True)
    buy_ema_rel_3 = DecimalParameter(0.97, 0.999, default=0.986, space='buy', decimals=3, optimize=False, load=True)

    buy_bb20_close_bblowerband_4 = DecimalParameter(0.96, 0.99, default=0.979, space='buy', optimize=False, load=True)
    buy_bb20_volume_4 = DecimalParameter(1.0, 20.0, default=10.0, space='buy', decimals=2, optimize=False, load=True)

    buy_ema_open_mult_5 = DecimalParameter(0.016, 0.03, default=0.019, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_5 = DecimalParameter(0.98, 1.0, default=0.999, space='buy', decimals=3, optimize=False, load=True)
    buy_ema_rel_5 = DecimalParameter(0.97, 0.999, default=0.982, space='buy', decimals=3, optimize=False, load=True)

    buy_ema_open_mult_6 = DecimalParameter(0.02, 0.03, default=0.025, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_6 = DecimalParameter(0.98, 0.999, default=0.984, space='buy', decimals=3, optimize=False, load=True)

    buy_volume_7 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_open_mult_7 = DecimalParameter(0.02, 0.04, default=0.03, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_7 = DecimalParameter(24.0, 50.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_rel_7 = DecimalParameter(0.97, 0.999, default=0.986, space='buy', decimals=3, optimize=False, load=True)

    buy_volume_8 = DecimalParameter(1.0, 6.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_8 = DecimalParameter(36.0, 40.0, default=20.0, space='buy', decimals=1, optimize=False, load=True)
    buy_tail_diff_8 = DecimalParameter(3.0, 10.0, default=3.5, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_9 = DecimalParameter(1.0, 4.0, default=1.0, space='buy', decimals=2, optimize=False, load=True)
    buy_ma_offset_9 = DecimalParameter(0.94, 0.99, default=0.97, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_9 = DecimalParameter(0.97, 0.99, default=0.985, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_9 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_9 = DecimalParameter(70.0, 90.0, default=88.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_9 = DecimalParameter(36.0, 65.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_10 = DecimalParameter(1.0, 8.0, default=2.4, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_10 = DecimalParameter(0.93, 0.97, default=0.944, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_10 = DecimalParameter(0.97, 0.99, default=0.994, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_10 = DecimalParameter(20.0, 40.0, default=37.0, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_11 = DecimalParameter(0.93, 0.99, default=0.939, space='buy', decimals=3, optimize=False, load=True)
    buy_min_inc_11 = DecimalParameter(0.005, 0.05, default=0.022, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_11 = DecimalParameter(40.0, 60.0, default=56.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_11 = DecimalParameter(70.0, 90.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_11 = DecimalParameter(30.0, 48.0, default=48.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_11 = DecimalParameter(36.0, 56.0, default=38.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_12 = DecimalParameter(1.0, 10.0, default=1.7, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_12 = DecimalParameter(0.93, 0.97, default=0.936, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_12 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ewo_12 = DecimalParameter(2.0, 6.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_13 = DecimalParameter(1.0, 10.0, default=1.6, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_13 = DecimalParameter(0.93, 0.98, default=0.978, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_13 = DecimalParameter(-14.0, -7.0, default=-10.4, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_14 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_open_mult_14 = DecimalParameter(0.01, 0.03, default=0.014, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_14 = DecimalParameter(0.98, 1.0, default=0.986, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_14 = DecimalParameter(0.93, 0.99, default=0.97, space='buy', decimals=3, optimize=False, load=True)

    buy_volume_15 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_open_mult_15 = DecimalParameter(0.02, 0.04, default=0.018, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_15 = DecimalParameter(0.93, 0.99, default=0.954, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_15 = DecimalParameter(30.0, 50.0, default=28.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_rel_15 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=False, load=True)

    buy_volume_16 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_16 = DecimalParameter(0.93, 0.97, default=0.952, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_16 = DecimalParameter(26.0, 50.0, default=31.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ewo_16 = DecimalParameter(4.0, 8.0, default=2.8, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_17 = DecimalParameter(0.5, 8.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_17 = DecimalParameter(0.93, 0.98, default=0.958, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_17 = DecimalParameter(-18.0, -10.0, default=-12.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_18 = DecimalParameter(1.0, 6.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_18 = DecimalParameter(16.0, 32.0, default=26.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_18 = DecimalParameter(0.98, 1.0, default=0.982, space='buy', decimals=3, optimize=False, load=True)

    buy_rsi_1h_min_19 = DecimalParameter(40.0, 70.0, default=50.0, space='buy', decimals=1, optimize=True, load=True)
    buy_chop_min_19 = DecimalParameter(20.0, 60.0, default=24.1, space='buy', decimals=1, optimize=True, load=True)

    buy_volume_20 = DecimalParameter(0.5, 6.0, default=1.2, space='buy', decimals=1, optimize=False, load=True)
    #buy_ema_rel_20 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_20 = DecimalParameter(20.0, 36.0, default=26.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_20 = DecimalParameter(14.0, 30.0, default=20.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_21 = DecimalParameter(0.5, 6.0, default=3.0, space='buy', decimals=1, optimize=False, load=True)
    #buy_ema_rel_21 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_21 = DecimalParameter(10.0, 28.0, default=23.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_21 = DecimalParameter(18.0, 40.0, default=24.0, space='buy', decimals=1, optimize=False, load=True)

    # Sell

    sell_condition_1_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_5_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_6_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_7_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_8_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)

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
    sell_custom_rsi_0 = DecimalParameter(30.0, 40.0, default=33.0, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_profit_1 = DecimalParameter(0.01, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_1 = DecimalParameter(30.0, 50.0, default=38.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_2 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_2 = DecimalParameter(34.0, 50.0, default=43.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_3 = DecimalParameter(0.06, 0.30, default=0.08, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_3 = DecimalParameter(38.0, 55.0, default=48.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_4 = DecimalParameter(0.3, 0.6, default=0.25, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_4 = DecimalParameter(40.0, 58.0, default=50.0, space='sell', decimals=2, optimize=False, load=True)

    sell_custom_under_profit_1 = DecimalParameter(0.01, 0.10, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_1 = DecimalParameter(36.0, 60.0, default=56.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_2 = DecimalParameter(0.01, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_2 = DecimalParameter(46.0, 66.0, default=60.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_3 = DecimalParameter(0.01, 0.10, default=0.6, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_3 = DecimalParameter(50.0, 68.0, default=62.0, space='sell', decimals=1, optimize=False, load=True)

    sell_custom_dec_profit_1 = DecimalParameter(0.01, 0.10, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_dec_profit_2 = DecimalParameter(0.05, 0.2, default=0.07, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.15, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.46, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.2, default=0.18, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_2 = DecimalParameter(0.01, 0.1, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.12, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.14, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_3 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_3 = DecimalParameter(0.08, 0.16, default=0.1, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_3 = DecimalParameter(0.01, 0.04, default=0.01, space='sell', decimals=3, optimize=False, load=True)

    sell_custom_profit_under_rel_1 = DecimalParameter(0.01, 0.04, default=0.024, space='sell', optimize=False, load=True)
    sell_custom_profit_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)

    sell_custom_stoploss_under_rel_1 = DecimalParameter(0.001, 0.02, default=0.004, space='sell', optimize=False, load=True)
    sell_custom_stoploss_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=8.0, space='sell', optimize=False, load=True)

    #############################################################

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])


    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)

        if (last_candle is not None):
            if (current_profit > self.sell_custom_profit_4.value) & (last_candle['rsi'] < self.sell_custom_rsi_4.value):
                return 'signal_profit_4'
            elif (current_profit > self.sell_custom_profit_3.value) & (last_candle['rsi'] < self.sell_custom_rsi_3.value):
                return 'signal_profit_3'
            elif (current_profit > self.sell_custom_profit_2.value) & (last_candle['rsi'] < self.sell_custom_rsi_2.value):
                return 'signal_profit_2'
            elif (current_profit > self.sell_custom_profit_1.value) & (last_candle['rsi'] < self.sell_custom_rsi_1.value):
                return 'signal_profit_1'
            elif (current_profit > self.sell_custom_profit_0.value) & (last_candle['rsi'] < self.sell_custom_rsi_0.value):
                return 'signal_profit_0'

            elif (current_profit > self.sell_custom_under_profit_1.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_1.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_1'
            elif (current_profit > self.sell_custom_under_profit_2.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_2.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_2'
            elif (current_profit > self.sell_custom_under_profit_3.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_3.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_3'

            elif (current_profit > self.sell_custom_dec_profit_1.value) & (last_candle['sma_200_dec']):
                return 'signal_profit_d_1'
            elif (current_profit > self.sell_custom_dec_profit_2.value) & (last_candle['close'] < last_candle['ema_100']):
                return 'signal_profit_d_2'

            elif (current_profit > self.sell_trail_profit_min_1.value) & (current_profit < self.sell_trail_profit_max_1.value) & (max_profit > (current_profit + self.sell_trail_down_1.value)):
                return 'signal_profit_t_1'
            elif (current_profit > self.sell_trail_profit_min_2.value) & (current_profit < self.sell_trail_profit_max_2.value) & (max_profit > (current_profit + self.sell_trail_down_2.value)):
                return 'signal_profit_t_2'

            elif (last_candle['close'] < last_candle['ema_200']) & (current_profit > self.sell_trail_profit_min_3.value) & (current_profit < self.sell_trail_profit_max_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)):
                return 'signal_profit_u_t_1'

            elif (current_profit > 0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_profit_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_profit_under_rsi_diff_1.value):
                return 'signal_profit_u_e_1'

            elif (current_profit < -0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_stoploss_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_under_rsi_diff_1.value):
                return 'signal_stoploss_u_1'

        return None

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
        informative_1h['ema_15'] = ta.EMA(informative_1h, timeperiod=15)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # SMA
        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)
        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)
        # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband'] = bollinger['lower']
        informative_1h['bb_middleband'] = bollinger['mid']
        informative_1h['bb_upperband'] = bollinger['upper']
        # Pump protections
        informative_1h['safe_pump_24'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_1.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / self.buy_pump_pull_threshold_1.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))
        informative_1h['safe_pump_36'] = ((((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / informative_1h['close'].rolling(36).min()) < self.buy_pump_threshold_2.value) | (((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / self.buy_pump_pull_threshold_2.value) > (informative_1h['close'] - informative_1h['close'].rolling(36).min())))
        informative_1h['safe_pump_48'] = ((((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / informative_1h['close'].rolling(48).min()) < self.buy_pump_threshold_3.value) | (((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / self.buy_pump_pull_threshold_3.value) > (informative_1h['close'] - informative_1h['close'].rolling(48).min())))

        informative_1h['safe_pump_24_strict'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_4.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / self.buy_pump_pull_threshold_4.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))
        informative_1h['safe_pump_36_strict'] = ((((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / informative_1h['close'].rolling(36).min()) < self.buy_pump_threshold_5.value) | (((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / self.buy_pump_pull_threshold_5.value) > (informative_1h['close'] - informative_1h['close'].rolling(36).min())))
        informative_1h['safe_pump_48_strict'] = ((((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / informative_1h['close'].rolling(48).min()) < self.buy_pump_threshold_6.value) | (((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / self.buy_pump_pull_threshold_6.value) > (informative_1h['close'] - informative_1h['close'].rolling(48).min())))

        informative_1h['safe_pump_24_loose'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_7.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / self.buy_pump_pull_threshold_7.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))
        informative_1h['safe_pump_36_loose'] = ((((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / informative_1h['close'].rolling(36).min()) < self.buy_pump_threshold_8.value) | (((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / self.buy_pump_pull_threshold_8.value) > (informative_1h['close'] - informative_1h['close'].rolling(36).min())))
        informative_1h['safe_pump_48_loose'] = ((((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / informative_1h['close'].rolling(48).min()) < self.buy_pump_threshold_9.value) | (((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / self.buy_pump_pull_threshold_9.value) > (informative_1h['close'] - informative_1h['close'].rolling(48).min())))

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
        dataframe['ewo'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Chopiness
        dataframe['chop']= qtpylib.chopiness(dataframe, 14)

        # Dip protection
        dataframe['safe_dips'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
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
                (dataframe['sma_200'] > dataframe['sma_200'].shift(50)) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_24_1h']) &

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

                (dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(50)) &

                (dataframe['safe_pump_24_strict_1h']) &

                (dataframe['volume_mean_4'] * self.buy_volume_2.value > dataframe['volume']) &

                #(dataframe['rsi_1h'] > self.buy_rsi_1h_min_2.value) &
                #(dataframe['rsi_1h'] < self.buy_rsi_1h_max_2.value) &
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
                (dataframe['ema_100_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_pump_36_strict_1h']) &

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

                (dataframe['safe_dips']) &
                (dataframe['safe_pump_36_strict_1h']) &

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

                (dataframe['ema_100_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips_loose']) &
                (dataframe['safe_pump_36_strict_1h']) &

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

                (dataframe['safe_dips_strict']) &

                (dataframe['volume'].rolling(4).mean() * self.buy_volume_7.value > dataframe['volume']) &

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

                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips_loose']) &
                (dataframe['safe_pump_24_1h']) &

                (dataframe['rsi'] < self.buy_rsi_8.value) &
                (dataframe['volume'] > (dataframe['volume'].shift(1) * self.buy_volume_8.value)) &
                (dataframe['close'] > dataframe['open']) &
                ((dataframe['close'] - dataframe['low']) > ((dataframe['close'] - dataframe['open']) * self.buy_tail_diff_8.value)) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_9_enable.value &

                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_100'] > dataframe['ema_200']) &

                (dataframe['safe_dips_strict']) &
                (dataframe['safe_pump_24_loose_1h']) &

                (dataframe['volume_mean_4'] * self.buy_volume_9.value > dataframe['volume']) &

                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_9.value) &
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

                (dataframe['safe_dips_loose']) &
                (dataframe['safe_pump_24_loose_1h']) &

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

                (dataframe['safe_dips_loose']) &
                (dataframe['safe_pump_24_loose_1h']) &
                (dataframe['safe_pump_36_1h']) &
                (dataframe['safe_pump_48_loose_1h']) &

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
                (dataframe['safe_pump_24_1h']) &

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
                (dataframe['safe_pump_24_loose_1h']) &
                (dataframe['safe_pump_36_loose_1h']) &

                ((dataframe['volume_mean_4'] * self.buy_volume_13.value) > dataframe['volume']) &

                (dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_13.value) &
                (dataframe['ewo'] < self.buy_ewo_13.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_14_enable.value &

                (dataframe['sma_200'] > dataframe['sma_200'].shift(30)) &
                (dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(50)) &

                (dataframe['safe_dips_loose']) &
                (dataframe['safe_pump_24_1h']) &

                (dataframe['volume_mean_4'] * self.buy_volume_14.value > dataframe['volume']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_14.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_bb_offset_14.value)) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_14.value) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_15_enable.value &

                (dataframe['close'] > dataframe['ema_200_1h'] * self.buy_ema_rel_15.value) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips']) &
                (dataframe['safe_pump_36_strict_1h']) &

                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_15.value)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['rsi'] < self.buy_rsi_15.value) &
                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_15.value) &

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
                (dataframe['safe_pump_24_loose_1h']) &

                ((dataframe['volume_mean_4'] * self.buy_volume_17.value) > dataframe['volume']) &

                (dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_17.value) &
                (dataframe['ewo'] < self.buy_ewo_17.value) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_18_enable.value &

                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['ema_100'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &
                (dataframe['sma_200'] > dataframe['sma_200'].shift(20)) &
                (dataframe['sma_200'] > dataframe['sma_200'].shift(44)) &
                (dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(36)) &
                (dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(72)) &

                (dataframe['safe_dips']) &
                (dataframe['safe_pump_24_strict_1h']) &

                ((dataframe['volume_mean_4'] * self.buy_volume_18.value) > dataframe['volume']) &

                (dataframe['rsi'] < self.buy_rsi_18.value) &
                (dataframe['close'] < (dataframe['bb_lowerband'] * self.buy_bb_offset_18.value)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_19_enable.value &

                (dataframe['ema_100_1h'] > dataframe['ema_200_1h']) &

                (dataframe['sma_200'] > dataframe['sma_200'].shift(36)) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips']) &
                (dataframe['safe_pump_24_1h']) &

                (dataframe['close'].shift(1) > dataframe['ema_100_1h']) &
                (dataframe['low'] < dataframe['ema_100_1h']) &
                (dataframe['close'] > dataframe['ema_100_1h']) &
                (dataframe['rsi_1h'] > self.buy_rsi_1h_min_19.value) &
                (dataframe['chop'] < self.buy_chop_min_19.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_20_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips']) &
                (dataframe['safe_pump_24_loose_1h']) &

                ((dataframe['volume_mean_4'] * self.buy_volume_20.value) > dataframe['volume']) &

                (dataframe['rsi'] < self.buy_rsi_20.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_20.value) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.buy_condition_21_enable.value &

                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips_strict']) &

                ((dataframe['volume_mean_4'] * self.buy_volume_21.value) > dataframe['volume']) &

                (dataframe['rsi'] < self.buy_rsi_21.value) &
                (dataframe['rsi_1h'] < self.buy_rsi_1h_21.value) &

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

        """
	for i in self.ma_types:
            conditions.append(
                (
                    (dataframe['close'] > dataframe[f'{i}_offset_sell']) &
                    (dataframe['volume'] > 0)
                )
        )
	"""

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
