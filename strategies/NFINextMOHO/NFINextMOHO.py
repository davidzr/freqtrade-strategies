import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, timeframe_to_minutes
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from technical.util import resample_to_interval, resampled_merge
from technical.indicators import zema


###########################################################################################################
##                NostalgiaForInfinityV8 by iterativ                                                     ##
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


class NFINextMOHO(IStrategy):
    INTERFACE_VERSION = 2

    # # ROI table:
    minimal_roi = {
        "0": 10,
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
    res_timeframe = '30m'
    info_timeframe = '1h'

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
    sell_custom_rsi_6 = DecimalParameter(38.0, 55.0, default=52.0, space='sell', decimals=2, optimize=False, load=True)
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
    sell_custom_under_rsi_0 = DecimalParameter(28.0, 40.0, default=38.0, space='sell', decimals=1, optimize=False, load=True)
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

    # Trail 3
    sell_trail_profit_min_4 = DecimalParameter(0.01, 0.12, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_4 = DecimalParameter(0.02, 0.1, default=0.06, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_4 = DecimalParameter(0.01, 0.06, default=0.02, space='sell', decimals=3, optimize=False, load=True)

    # Under & near EMA200, accept profit
    sell_custom_profit_under_rel_1 = DecimalParameter(0.01, 0.04, default=0.024, space='sell', optimize=False, load=True)
    sell_custom_profit_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)

    # Under & near EMA200, take the loss
    sell_custom_stoploss_under_rel_1 = DecimalParameter(0.001, 0.02, default=0.002, space='sell', optimize=False, load=True)
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
    sell_custom_stoploss_pump_max_profit_1 = DecimalParameter(0.01, 0.04, default=0.01, space='sell', decimals=3, optimize=False, load=True)
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
            if (self.sell_custom_profit_8.value > current_profit > self.sell_custom_profit_7.value) & (last_candle['rsi'] < self.sell_custom_rsi_7.value) & (last_candle['cmf'] < 0.0):
                return 'signal_profit_7'
            if (self.sell_custom_profit_7.value > current_profit > self.sell_custom_profit_6.value) & (last_candle['rsi'] < self.sell_custom_rsi_6.value) & (last_candle['cmf'] < 0.0):
                return 'signal_profit_6'
            if (self.sell_custom_profit_6.value > current_profit > self.sell_custom_profit_5.value) & (last_candle['rsi'] < self.sell_custom_rsi_5.value) & (last_candle['cmf'] < 0.0):
                return 'signal_profit_5'
            elif (self.sell_custom_profit_5.value > current_profit > self.sell_custom_profit_4.value) & (last_candle['rsi'] < self.sell_custom_rsi_4.value) & (last_candle['cmf'] < 0.0):
                return 'signal_profit_4'
            elif (self.sell_custom_profit_4.value > current_profit > self.sell_custom_profit_3.value) & (last_candle['rsi'] < self.sell_custom_rsi_3.value) & (last_candle['cmf'] < 0.0):
                return 'signal_profit_3'
            elif (self.sell_custom_profit_3.value > current_profit > self.sell_custom_profit_2.value) & (last_candle['rsi'] < self.sell_custom_rsi_2.value) & (last_candle['cmf'] < 0.0):
                return 'signal_profit_2'
            elif (self.sell_custom_profit_2.value > current_profit > self.sell_custom_profit_1.value) & (last_candle['rsi'] < self.sell_custom_rsi_1.value) & (last_candle['cmf'] < 0.0):
                return 'signal_profit_1'
            elif (self.sell_custom_profit_1.value > current_profit > self.sell_custom_profit_0.value) & (last_candle['rsi'] < self.sell_custom_rsi_0.value) & (last_candle['cmf'] < 0.0):
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
            elif (self.sell_custom_under_profit_1.value > current_profit > self.sell_custom_under_profit_0.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_0.value) & (last_candle['close'] < last_candle['ema_200']) & (last_candle['cmf'] < 0.0):
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
            elif (self.sell_trail_profit_max_4.value > current_profit > self.sell_trail_profit_min_4.value) & (max_profit > (current_profit + self.sell_trail_down_4.value)) & (last_candle['sma_200_dec_24']) & (last_candle['cmf'] < 0.0):
                return 'signal_profit_t_4'

            elif (last_candle['close'] < last_candle['ema_200']) & (current_profit > self.sell_trail_profit_min_3.value) & (current_profit < self.sell_trail_profit_max_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)):
                return 'signal_profit_u_t_1'

            # elif (last_candle['sell_pump_24_1_1h']) & (0.1 > current_profit > 0.07) & (last_candle['rsi'] < 56.0) & (current_time - timedelta(minutes=20) < trade.open_date_utc):
            #     return 'signal_profit_p_s_1'

            elif (current_profit > 0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_profit_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_profit_under_rsi_diff_1.value):
                return 'signal_profit_u_e_1'

            elif (current_profit < -0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_stoploss_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_under_rsi_diff_1.value) & (last_candle['cmf'] < 0.0) & (last_candle['sma_200_dec_24']) & (current_time - timedelta(minutes=720) > trade.open_date_utc):
                return 'signal_stoploss_u_1'

            elif (self.sell_custom_stoploss_long_profit_min_1.value < current_profit < self.sell_custom_stoploss_long_profit_max_1.value) & (current_profit > (-max_loss + self.sell_custom_stoploss_long_recover_1.value)) & (last_candle['cmf'] < 0.0) & (last_candle['close'] < last_candle['ema_200'])  & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_long_rsi_diff_1.value) & (last_candle['sma_200_dec_24']) & (current_time - timedelta(minutes=1200) > trade.open_date_utc):
                return 'signal_stoploss_l_r_u_1'

            elif (current_profit < -0.0) & (current_profit > (-max_loss + self.sell_custom_stoploss_long_recover_2.value)) & (last_candle['close'] < last_candle['ema_200']) & (last_candle['cmf'] < 0.0) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_long_rsi_diff_2.value) & (last_candle['sma_200_dec_24']) & (current_time - timedelta(minutes=1200) > trade.open_date_utc):
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

            # elif (max_profit < self.sell_custom_stoploss_pump_max_profit_1.value) & (self.sell_custom_stoploss_pump_min_1.value < current_profit < self.sell_custom_stoploss_pump_max_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['cmf'] < 0.0) & (last_candle['sma_200_dec_20']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_1.value)):
            #     return 'signal_stoploss_p_1'

            elif (max_profit < self.sell_custom_stoploss_pump_max_profit_2.value) & (current_profit < self.sell_custom_stoploss_pump_loss_2.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['cmf'] < 0.0) & (last_candle['sma_200_dec_20_1h']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_2.value)):
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

    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.info_timeframe) for pair in pairs]
        informative_pairs.append(('BTC/USDT', self.timeframe))
        informative_pairs.append(('BTC/USDT', self.info_timeframe))
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe)

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
        informative_1h['sma_200_dec_20'] = informative_1h['sma_200'] < informative_1h['sma_200'].shift(20)

        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb20_2_low'] = bollinger['lower']
        informative_1h['bb20_2_mid'] = bollinger['mid']
        informative_1h['bb20_2_upp'] = bollinger['upper']
        informative_1h['bb_lowerband'] = bollinger['lower']
        informative_1h['bb_middleband'] = bollinger['mid']
        informative_1h['bb_upperband'] = bollinger['upper']

        # Pump protections NFI5
        informative_1h['safe_pump_24'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_1.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / self.buy_pump_pull_threshold_1.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))
        informative_1h['safe_pump_36'] = ((((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / informative_1h['close'].rolling(36).min()) < self.buy_pump_threshold_2.value) | (((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / self.buy_pump_pull_threshold_2.value) > (informative_1h['close'] - informative_1h['close'].rolling(36).min())))
        informative_1h['safe_pump_48'] = ((((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / informative_1h['close'].rolling(48).min()) < self.buy_pump_threshold_3.value) | (((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / self.buy_pump_pull_threshold_3.value) > (informative_1h['close'] - informative_1h['close'].rolling(48).min())))

        informative_1h['safe_pump_24_strict'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_4.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / self.buy_pump_pull_threshold_4.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))
        informative_1h['safe_pump_36_strict'] = ((((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / informative_1h['close'].rolling(36).min()) < self.buy_pump_threshold_5.value) | (((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / self.buy_pump_pull_threshold_5.value) > (informative_1h['close'] - informative_1h['close'].rolling(36).min())))
        informative_1h['safe_pump_48_strict'] = ((((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / informative_1h['close'].rolling(48).min()) < self.buy_pump_threshold_6.value) | (((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / self.buy_pump_pull_threshold_6.value) > (informative_1h['close'] - informative_1h['close'].rolling(48).min())))

        informative_1h['safe_pump_24_loose'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / informative_1h['close'].rolling(24).min()) < self.buy_pump_threshold_7.value) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / self.buy_pump_pull_threshold_7.value) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))
        informative_1h['safe_pump_36_loose'] = ((((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / informative_1h['close'].rolling(36).min()) < self.buy_pump_threshold_8.value) | (((informative_1h['open'].rolling(36).max() - informative_1h['close'].rolling(36).min()) / self.buy_pump_pull_threshold_8.value) > (informative_1h['close'] - informative_1h['close'].rolling(36).min())))
        informative_1h['safe_pump_48_loose'] = ((((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / informative_1h['close'].rolling(48).min()) < self.buy_pump_threshold_9.value) | (((informative_1h['open'].rolling(48).max() - informative_1h['close'].rolling(48).min()) / self.buy_pump_pull_threshold_9.value) > (informative_1h['close'] - informative_1h['close'].rolling(48).min())))

        # Chaikin Money Flow
        informative_1h['cmf'] = chaikin_money_flow(informative_1h, 20)

        # Pump protections
        informative_1h['hl_pct_change_48'] = self.range_percent_change(informative_1h, 'HL', 48)
        informative_1h['hl_pct_change_36'] = self.range_percent_change(informative_1h, 'HL', 36)
        informative_1h['hl_pct_change_24'] = self.range_percent_change(informative_1h, 'HL', 24)

        informative_1h['oc_pct_change_48'] = self.range_percent_change(informative_1h, 'OC', 48)
        informative_1h['oc_pct_change_36'] = self.range_percent_change(informative_1h, 'OC', 36)
        informative_1h['oc_pct_change_24'] = self.range_percent_change(informative_1h, 'OC', 24)

        informative_1h['sell_pump_48_1'] = (informative_1h['hl_pct_change_48'] > self.sell_pump_threshold_48_1.value)
        informative_1h['sell_pump_48_2'] = (informative_1h['hl_pct_change_48'] > self.sell_pump_threshold_48_2.value)
        informative_1h['sell_pump_48_3'] = (informative_1h['hl_pct_change_48'] > self.sell_pump_threshold_48_3.value)

        informative_1h['sell_pump_36_1'] = (informative_1h['hl_pct_change_36'] > self.sell_pump_threshold_36_1.value)
        informative_1h['sell_pump_36_2'] = (informative_1h['hl_pct_change_36'] > self.sell_pump_threshold_36_2.value)
        informative_1h['sell_pump_36_3'] = (informative_1h['hl_pct_change_36'] > self.sell_pump_threshold_36_3.value)

        informative_1h['sell_pump_24_1'] = (informative_1h['hl_pct_change_24'] > self.sell_pump_threshold_24_1.value)
        informative_1h['sell_pump_24_2'] = (informative_1h['hl_pct_change_24'] > self.sell_pump_threshold_24_2.value)
        informative_1h['sell_pump_24_3'] = (informative_1h['hl_pct_change_24'] > self.sell_pump_threshold_24_3.value)

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # BB 40 - STD2
        bb_40_std2 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['bb40_2_low']= bb_40_std2['lower']
        dataframe['bb40_2_mid'] = bb_40_std2['mid']
        dataframe['bb40_2_delta'] = (bb_40_std2['mid'] - dataframe['bb40_2_low']).abs()
        dataframe['lower'] = bb_40_std2['lower']
        dataframe['mid'] = bb_40_std2['mid']
        dataframe['bbdelta'] = (bb_40_std2['mid'] - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['bb40_2_low']).abs()

        # BB 20 - STD2
        bb_20_std2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bb_20_std2['lower']
        dataframe['bb_middleband'] = bb_20_std2['mid']
        dataframe['bb_upperband'] = bb_20_std2['upper']
        dataframe['bb20_2_low'] = bb_20_std2['lower']
        dataframe['bb20_2_mid'] = bb_20_std2['mid']
        dataframe['bb20_2_upp'] = bb_20_std2['upper']

        # EMA 200
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_15'] = ta.EMA(dataframe, timeperiod=15)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_35'] = ta.EMA(dataframe, timeperiod=35)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['sma_200_dec'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        dataframe['sma_200_dec_20'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        dataframe['sma_200_dec_24'] = dataframe['sma_200'] < dataframe['sma_200'].shift(24)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # EWO
        dataframe['ewo'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_20'] = ta.RSI(dataframe, timeperiod=20)

        # Chopiness
        dataframe['chop']= qtpylib.chopiness(dataframe, 14)

        # Zero-Lag EMA
        dataframe['zema'] = zema(dataframe, period=61)

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

    def resampled_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: "btc_" + s  if (not s in ignore_columns) else s, inplace=True)

        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: "btc_" + s if (not s in ignore_columns) else s, inplace=True)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        '''
        --> BTC informative (5m/1h)
        ___________________________________________________________________________________________
        '''
        btc_base_tf = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [(s + "_" + self.timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_info_tf = self.dp.get_pair_dataframe("BTC/USDT", self.info_timeframe)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.info_timeframe, ffill=True)
        drop_columns = [(s + "_" + self.info_timeframe) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        '''
        --> Informative timeframe
        ___________________________________________________________________________________________
        '''
        # populate informative indicators
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        # Merge informative into dataframe
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.info_timeframe, ffill=True)
        drop_columns = [(s + "_" + self.info_timeframe) for s in ['date']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        '''
        --> Resampled to another timeframe
        ___________________________________________________________________________________________
        '''
        # resampled = resample_to_interval(dataframe, timeframe_to_minutes(self.res_timeframe))
        # resampled = self.resampled_tf_indicators(resampled, metadata)
        # # Merge resampled info dataframe
        # dataframe = resampled_merge(dataframe, resampled, fill_na=True)
        # dataframe.rename(columns=lambda s: s+"_{}".format(self.res_timeframe) if "resample_" in s else s, inplace=True)
        # dataframe.rename(columns=lambda s: s.replace("resample_{}_".format(self.res_timeframe.replace("m","")), ""), inplace=True)
        # drop_columns = [(s + "_" + self.res_timeframe) for s in ['date']]
        # dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        '''
        --> The indicators for the normal (5m) timeframe
        ___________________________________________________________________________________________
        '''
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
