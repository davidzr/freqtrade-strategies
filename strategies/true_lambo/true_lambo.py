# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, DatetimeIndex, merge
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from functools import reduce
from technical.indicators import RMI, zema

# --------------------------------
def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

# VWAP bands
def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def top_percent_change(dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price
        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

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

class true_lambo(IStrategy):

    '''
    @ jilv220

    Based on BB_RPB_3c.

    The discovery is that the pump protection is only needed for EWO level 2 (4 ~ 8)
    And the dump protection is only needed for EWO level 5 (< -8)

    Higher parameter delta is needed for EWO level 4 (-4 ~ -8)

    Vwap and Cluc can handle rest of the market pretty well

    '''

    ##########################################################################

    # Hyperopt result area

    # buy space
    buy_params = {
        ##
        "buy_pump_2_factor": 1.125,
        ##
        "buy_crash_4_tpct_0": 0.141,
        "buy_crash_4_tpct_3": 0.031,
        "buy_crash_4_tpct_9": 0.091,
        ##
        "buy_adx": 20,
        "buy_fastd": 20,
        "buy_fastk": 22,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 4.179,
        ##
        "buy_gumbo_ema": 1.121,
        "buy_gumbo_ewo_low": -9.442,
        "buy_gumbo_cti": -0.374,
        "buy_gumbo_r14": -51.971,
        "buy_gumbo_tpct_0": 0.067,
        "buy_gumbo_tpct_3": 0.0,
        "buy_gumbo_tpct_9": 0.016,
        ##
        "buy_clucha_bbdelta_close": 0.03734,
        "buy_clucha_bbdelta_close_2": 0.0391,
        "buy_clucha_bbdelta_close_3": 0.03943,
        "buy_clucha_bbdelta_close_4": 0.04202,
        ##
        "buy_clucha_bbdelta_tail": 0.78624,
        "buy_clucha_bbdelta_tail_2": 1.05718,
        "buy_clucha_bbdelta_tail_3": 0.77701,
        "buy_clucha_bbdelta_tail_4": 0.418,
        ##
        "buy_clucha_closedelta_close": 0.01382,
        "buy_clucha_closedelta_close_2": 0.01092,
        "buy_clucha_closedelta_close_3": 0.00935,
        "buy_clucha_closedelta_close_4": 0.01234,
        ##
        "buy_clucha_rocr_1h": 0.70818,
        "buy_clucha_rocr_1h_2": 0.15848,
        "buy_clucha_rocr_1h_3": 0.88989,
        "buy_clucha_rocr_1h_4": 0.99234,
        ##
        "buy_vwap_closedelta": 19.889,
        "buy_vwap_closedelta_2": 20.099,
        "buy_vwap_closedelta_3": 27.526,
        "buy_vwap_closedelta_4": 22.26,
        "buy_vwap_closedelta_5": 23.227,
        ##
        "buy_vwap_cti": -0.258,
        "buy_vwap_cti_2": -0.748,
        "buy_vwap_cti_3": -0.227,
        "buy_vwap_cti_4": -0.763,
        "buy_vwap_cti_5": -0.021,
        ##
        "buy_vwap_width": 0.266,
        "buy_vwap_width_2": 3.212,
        "buy_vwap_width_3": 0.47,
        "buy_vwap_width_4": 9.76,
        "buy_vwap_width_5": 5.374,
        ##
        "buy_lambo2_ema": 0.986,
        "buy_lambo2_rsi14": 32,
        "buy_lambo2_rsi4": 30,
        ##
        "buy_V_bb_width": 0.03,
        "buy_V_bb_width_2": 0.08,
        "buy_V_bb_width_3": 0.097,
        "buy_V_bb_width_4": 0.054,
        "buy_V_bb_width_5": 0.063,
        ##
        "buy_V_cti": -0.872,
        "buy_V_cti_2": -0.842,
        "buy_V_cti_3": -0.888,
        "buy_V_cti_4": -0.57,
        "buy_V_cti_5": -0.086,
        ##
        "buy_V_mfi": 35.342,
        "buy_V_mfi_2": 24.775,
        "buy_V_mfi_3": 17.053,
        "buy_V_mfi_4": 13.907,
        "buy_V_mfi_5": 38.158,
        ##
        "buy_V_r14": -67.209,
        "buy_V_r14_2": -6.723,
        "buy_V_r14_3": -68.059,
        "buy_V_r14_4": -88.943,
        "buy_V_r14_5": -41.493,
    }

    # sell space
    sell_params = {
        ##
        "base_nb_candles_sell": 23,
        "high_offset": 0.87,
        "high_offset_2": 1.166,
        ##
        "high_offset": 0.963,
        "high_offset_2": 1.391,
        "pHSL": -0.083,
        "pPF_1": 0.011,
        "pPF_2": 0.048,
        "pSL_1": 0.02,
        "pSL_2": 0.041,
    }

    # ROI
    minimal_roi = {
        "0": 0.092,
        "29": 0.042,
        "85": 0.028,
        "128": 0.005
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    # Disabled
    stoploss = -0.99

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Custom stoploss
    use_custom_stoploss = True
    use_sell_signal = True
    startup_candle_count: int = 400

    ############################################################################

    ## Buy params

    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 45, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 45, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 45, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(-12, 12, default=3.553, optimize = is_optimize_cofi)

    is_optimize_clucha = False
    buy_clucha_bbdelta_close = DecimalParameter(0.0005, 0.042, default=0.034, decimals=5, optimize = is_optimize_clucha)
    buy_clucha_bbdelta_tail = DecimalParameter(0.7, 1.1, default=0.95, decimals=5, optimize = is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.0005, 0.025, default=0.019, decimals=5, optimize = is_optimize_clucha)
    buy_clucha_rocr_1h = DecimalParameter(0.001, 1.0, default=0.131, decimals=5, optimize = is_optimize_clucha)

    is_optimize_clucha_2 = False
    buy_clucha_bbdelta_close_2 = DecimalParameter(0.0005, 0.042, default=0.034, decimals=5, optimize = is_optimize_clucha_2)
    buy_clucha_bbdelta_tail_2 = DecimalParameter(0.7, 1.1, default=0.95, decimals=5, optimize = is_optimize_clucha_2)
    buy_clucha_closedelta_close_2 = DecimalParameter(0.0005, 0.025, default=0.019, decimals=5, optimize = is_optimize_clucha_2)
    buy_clucha_rocr_1h_2 = DecimalParameter(0.001, 1.0, default=0.131, decimals=5, optimize = is_optimize_clucha_2)

    is_optimize_clucha_3 = False
    buy_clucha_bbdelta_close_3 = DecimalParameter(0.0005, 0.042, default=0.034, decimals=5, optimize = is_optimize_clucha_3)
    buy_clucha_bbdelta_tail_3 = DecimalParameter(0.7, 1.1, default=0.95, decimals=5, optimize = is_optimize_clucha_3)
    buy_clucha_closedelta_close_3 = DecimalParameter(0.0005, 0.025, default=0.019, decimals=5, optimize = is_optimize_clucha_3)
    buy_clucha_rocr_1h_3 = DecimalParameter(0.001, 1.0, default=0.131, decimals=5, optimize = is_optimize_clucha_3)

    is_optimize_clucha_4 = True
    buy_clucha_bbdelta_close_4 = DecimalParameter(0.0005, 0.06, default=0.034, decimals=5, optimize = is_optimize_clucha_4)
    buy_clucha_bbdelta_tail_4 = DecimalParameter(0.35, 1.1, default=0.95, decimals=5, optimize = is_optimize_clucha_4)
    buy_clucha_closedelta_close_4 = DecimalParameter(0.0005, 0.025, default=0.019, decimals=5, optimize = is_optimize_clucha_4)
    buy_clucha_rocr_1h_4 = DecimalParameter(0.001, 1.0, default=0.131, decimals=5, optimize = is_optimize_clucha_4)

    is_optimize_gumbo = False
    buy_gumbo_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_gumbo)
    buy_gumbo_ewo_low = DecimalParameter(-12.0, 5, default=-5.585, optimize = is_optimize_gumbo)
    buy_gumbo_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_gumbo)
    buy_gumbo_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_gumbo)

    is_optimize_gumbo_protection = False
    buy_gumbo_tpct_0 = DecimalParameter(0.0, 0.25, default=0.131, decimals=2, optimize = is_optimize_gumbo_protection)
    buy_gumbo_tpct_3 = DecimalParameter(0.0, 0.25, default=0.131, decimals=2, optimize = is_optimize_gumbo_protection)
    buy_gumbo_tpct_9 = DecimalParameter(0.0, 0.25, default=0.131, decimals=2, optimize = is_optimize_gumbo_protection)

    is_optimize_vwap = False
    buy_vwap_width = DecimalParameter(0.05, 10.0, default=0.80 , optimize = is_optimize_vwap)
    buy_vwap_closedelta = DecimalParameter(10.0, 30.0, default=15.0, optimize = is_optimize_vwap)
    buy_vwap_cti = DecimalParameter(-0.9, -0.0, default=-0.6 , optimize = is_optimize_vwap)

    is_optimize_vwap_2 = False
    buy_vwap_width_2 = DecimalParameter(0.05, 10.0, default=0.80 , optimize = is_optimize_vwap_2)
    buy_vwap_closedelta_2 = DecimalParameter(10.0, 30.0, default=15.0, optimize = is_optimize_vwap_2)
    buy_vwap_cti_2 = DecimalParameter(-0.9, -0.0, default=-0.6 , optimize = is_optimize_vwap_2)

    is_optimize_vwap_3 = False
    buy_vwap_width_3 = DecimalParameter(0.05, 10.0, default=0.80 , optimize = is_optimize_vwap_3)
    buy_vwap_closedelta_3 = DecimalParameter(10.0, 30.0, default=15.0, optimize = is_optimize_vwap_3)
    buy_vwap_cti_3 = DecimalParameter(-0.9, -0.0, default=-0.6 , optimize = is_optimize_vwap_3)

    is_optimize_vwap_4 = True
    buy_vwap_width_4 = DecimalParameter(0.05, 10.0, default=0.80 , optimize = is_optimize_vwap_4)
    buy_vwap_closedelta_4 = DecimalParameter(10.0, 30.0, default=15.0, optimize = is_optimize_vwap_4)
    buy_vwap_cti_4 = DecimalParameter(-0.9, -0.0, default=-0.6 , optimize = is_optimize_vwap_4)

    is_optimize_lambo_2 = False
    buy_lambo2_ema = DecimalParameter(0.85, 1.15, default=0.942 , optimize = is_optimize_lambo_2)
    buy_lambo2_rsi4 = IntParameter(15, 45, default=45, optimize = is_optimize_lambo_2)
    buy_lambo2_rsi14 = IntParameter(15, 45, default=45, optimize = is_optimize_lambo_2)

    is_optimize_V = False
    buy_V_bb_width = DecimalParameter(0.01, 0.1, default=0.01 , optimize = is_optimize_V)
    buy_V_cti = DecimalParameter(-0.95, -0.5, default=-0.6 , optimize = is_optimize_V)
    buy_V_r14 = DecimalParameter(-100, 0, default=-60 , optimize = is_optimize_V)
    buy_V_mfi = DecimalParameter(10, 40, default=30 , optimize = is_optimize_V)

    is_optimize_V_2 = False
    buy_V_bb_width_2 = DecimalParameter(0.01, 0.1, default=0.01 , optimize = is_optimize_V_2)
    buy_V_cti_2 = DecimalParameter(-0.95, -0.5, default=-0.6 , optimize = is_optimize_V_2)
    buy_V_r14_2 = DecimalParameter(-100, 0, default=-60 , optimize = is_optimize_V_2)
    buy_V_mfi_2 = DecimalParameter(10, 40, default=30 , optimize = is_optimize_V_2)

    is_optimize_V_4 = False
    buy_V_bb_width_4 = DecimalParameter(0.01, 0.1, default=0.01 , optimize = is_optimize_V_4)
    buy_V_cti_4 = DecimalParameter(-0.95, -0.0, default=-0.6 , optimize = is_optimize_V_4)
    buy_V_r14_4 = DecimalParameter(-100, 0, default=-60 , optimize = is_optimize_V_4)
    buy_V_mfi_4 = DecimalParameter(10, 40, default=30 , optimize = is_optimize_V_4)

    is_optimize_V_5 = False
    buy_V_bb_width_5 = DecimalParameter(0.01, 0.1, default=0.01 , optimize = is_optimize_V_5)
    buy_V_cti_5 = DecimalParameter(-0.95, -0.0, default=-0.6 , optimize = is_optimize_V_5)
    buy_V_r14_5 = DecimalParameter(-100, 0, default=-60 , optimize = is_optimize_V_5)
    buy_V_mfi_5 = DecimalParameter(10, 40, default=30 , optimize = is_optimize_V_5)
    buy_V_diff_5 = DecimalParameter(0.10, 0.20, default=0.10 , optimize = is_optimize_V_5)

    is_optimize_pump_2 = False
    buy_pump_2_factor = DecimalParameter(1.0, 1.20, default= 1.1 , optimize = is_optimize_pump_2)

    is_optimize_crash_4 = False
    buy_crash_4_tpct_0 = DecimalParameter(0.02, 0.04, default=0.02, decimals=3, optimize = is_optimize_crash_4)
    buy_crash_4_tpct_3 = DecimalParameter(0.05, 0.10, default=0.05, decimals=2, optimize = is_optimize_crash_4)
    buy_crash_4_tpct_9 = DecimalParameter(0.13, 0.32, default=0.13, decimals=2, optimize = is_optimize_crash_4)

    ## Sell params
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    high_offset          = DecimalParameter(0.85, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2        = DecimalParameter(0.85, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)

    ## Trailing params

    # hard stoploss profit
    pHSL = DecimalParameter(-0.10, -0.040, default=-0.08, decimals=3, space='sell', load=True, optimize=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True, optimize=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True, optimize=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True, optimize=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True, optimize=True)

    ############################################################################

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        return informative_pairs

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
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

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1]
        #previous_candle_1 = dataframe.iloc[-2]
        #previous_candle_2 = dataframe.iloc[-3]

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)
        max_loss = ((trade.open_rate - trade.min_rate) / trade.min_rate)

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        buy_tags = buy_tag.split()

        # sell vwap (bear, conservative sell)
        if last_candle['ema_vwap_diff_50'] > 0.02:
            if current_profit >= 0.005:
                if (last_candle['close'] > last_candle['vwap_middleband'] * 0.99) and (last_candle['rsi'] > 41):
                    return f"sell_profit_vwap( {buy_tag})"

        return None

    ############################################################################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert self.dp, "DataProvider is required for multiple timeframes."

        # Bollinger bands (hyperopt hard to implement)
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        # BinH
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # SMA
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # EMA
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        dataframe['rsi_6'] = ta.RSI(dataframe, timeperiod=6)
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, 50, 200)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

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

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        # T3 Average
        dataframe['T3'] = T3(dataframe)

        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['vwap_width'] = ( (dataframe['vwap_upperband'] - dataframe['vwap_lowerband']) / dataframe['vwap_middleband'] ) * 100

        # Avg
        dataframe['average'] = (dataframe['close'] + dataframe['open'] + dataframe['high'] + dataframe['low']) / 4

        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)

        # Diff
        dataframe['ema_vwap_diff_50'] = ( ( dataframe['ema_50'] - dataframe['vwap_lowerband'] ) / dataframe['ema_50'] )

        # Crash protection
        dataframe['tpct_0'] = top_percent_change(dataframe , 0)
        dataframe['tpct_3'] = top_percent_change(dataframe , 3)
        dataframe['tpct_9'] = top_percent_change(dataframe , 9)
        #dataframe['tpct_24'] = top_percent_change(dataframe , 24)
        #dataframe['tpct_42'] = top_percent_change(dataframe , 42)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        ############################################################################

        # 1h tf
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        # Heikin Ashi
        inf_heikinashi = qtpylib.heikinashi(informative)
        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)

        # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bb_lowerband2'] = bollinger2['lower']
        informative['bb_middleband2'] = bollinger2['mid']
        informative['bb_upperband2'] = bollinger2['upper']
        informative['bb_width'] = ((informative['bb_upperband2'] - informative['bb_lowerband2']) / informative['bb_middleband2'])

        # T3 Average
        informative['T3'] = T3(informative)

        # RSI
        informative['rsi'] = ta.RSI(informative, timeperiod=14)

        # EMA
        informative['ema_200'] = ta.EMA(informative, timeperiod=200)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        is_pump_2 = (
                (dataframe['close'].rolling(48).max() >= (dataframe['close'] * self.buy_pump_2_factor.value ))
            )

        is_crash_4 = (
                (dataframe['tpct_0'] < self.buy_crash_4_tpct_0.value) &
                (dataframe['tpct_3'] < self.buy_crash_4_tpct_3.value) &
                (dataframe['tpct_9'] < self.buy_crash_4_tpct_9.value)
            )

        is_cofi = (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value)
            )

        is_gumbo = (                                                                        # Modified from gumbo1, creadit goes to original author @raph92
                (dataframe['bb_middleband2_1h'] >= dataframe['T3_1h']) &
                (dataframe['T3'] <= dataframe['ema_8'] * self.buy_gumbo_ema.value) &
                (dataframe['cti'] < self.buy_gumbo_cti.value) &
                (dataframe['r_14'] < self.buy_gumbo_r14.value) &
                (dataframe['EWO'] < self.buy_gumbo_ewo_low.value) &

                # Crash Protection
                (dataframe['tpct_0'] < self.buy_gumbo_tpct_0.value) &
                (dataframe['tpct_3'] < self.buy_gumbo_tpct_3.value) &
                (dataframe['tpct_9'] < self.buy_gumbo_tpct_9.value)
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
                ) &
                (dataframe['EWO'] > 8)
            )

        is_clucHA_2 = (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h_2.value ) &
                (
                        (dataframe['bb_lowerband2_40'].shift() > 0) &
                        (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close_2.value) &
                        (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close_2.value) &
                        (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail_2.value) &
                        (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                        (dataframe['ha_close'] < dataframe['ha_close'].shift())
                ) &
                (dataframe['EWO'] > 4) &
                (dataframe['EWO'] < 8) &
                (is_pump_2)
            )

        is_clucHA_3 = (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h_3.value ) &
                (
                        (dataframe['bb_lowerband2_40'].shift() > 0) &
                        (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close_3.value) &
                        (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close_3.value) &
                        (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail_3.value) &
                        (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                        (dataframe['ha_close'] < dataframe['ha_close'].shift())
                ) &
                (dataframe['EWO'] < 4) &
                (dataframe['EWO'] > -2.5)
            )

        is_clucHA_4 = (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h_4.value ) &
                (
                        (dataframe['bb_lowerband2_40'].shift() > 0) &
                        (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close_4.value) &
                        (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close_4.value) &
                        (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail_4.value) &
                        (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                        (dataframe['ha_close'] < dataframe['ha_close'].shift())
                ) &
                (dataframe['EWO'] < -4) &
                (dataframe['EWO'] > -8)
                #&
                #(is_crash_4)
            )

        is_vwap = (
                (dataframe['close'] < dataframe['vwap_lowerband']) &
                (dataframe['vwap_width'] > self.buy_vwap_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_vwap_closedelta.value / 1000 ) &
                (dataframe['cti'] < self.buy_vwap_cti.value) &
                (dataframe['EWO'] > 8)
            )

        is_vwap_2 = (
                (dataframe['close'] < dataframe['vwap_lowerband']) &
                (dataframe['vwap_width'] > self.buy_vwap_width_2.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_vwap_closedelta_2.value / 1000 ) &
                (dataframe['cti'] < self.buy_vwap_cti_2.value) &
                (dataframe['EWO'] > 4) &
                (dataframe['EWO'] < 8) &
                (is_pump_2)
            )

        is_vwap_3 = (
                (dataframe['close'] < dataframe['vwap_lowerband']) &
                (dataframe['vwap_width'] > self.buy_vwap_width_3.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_vwap_closedelta_3.value / 1000 ) &
                (dataframe['cti'] < self.buy_vwap_cti_3.value) &
                (dataframe['EWO'] < 4) &
                (dataframe['EWO'] > -2.5)
            )

        is_vwap_4 = (
                (dataframe['close'] < dataframe['vwap_lowerband']) &
                (dataframe['vwap_width'] > self.buy_vwap_width_4.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_vwap_closedelta_4.value / 1000 ) &
                (dataframe['cti'] < self.buy_vwap_cti_4.value) &
                (dataframe['EWO'] < -4) &
                (dataframe['EWO'] > -8)
                #&
                #(is_crash_4)
            )

        is_lambo_2 = (
                (dataframe['close'] < dataframe['ema_14'] * self.buy_lambo2_ema.value) &
                (dataframe['rsi_fast'] < self.buy_lambo2_rsi4.value) &
                (dataframe['rsi'] < self.buy_lambo2_rsi14.value) &
                (dataframe['EWO'] > 4) &
                (dataframe['EWO'] < 8) &
                (is_pump_2)
            )

        is_V = (
                (dataframe['bb_width'] > self.buy_V_bb_width.value) &
                (dataframe['cti'] < self.buy_V_cti.value) &
                (dataframe['r_14'] < self.buy_V_r14.value) &
                (dataframe['mfi'] < self.buy_V_mfi.value) &
                (dataframe['EWO'] > 8)
            )

        is_V_2 = (
                (dataframe['bb_width'] > self.buy_V_bb_width_2.value) &
                (dataframe['cti'] < self.buy_V_cti_2.value) &
                (dataframe['r_14'] < self.buy_V_r14_2.value) &
                (dataframe['mfi'] < self.buy_V_mfi_2.value) &
                (dataframe['EWO'] > 4) &
                (dataframe['EWO'] < 8) &
                (is_pump_2)
            )

        is_V_5 = (
                (dataframe['bb_width'] > self.buy_V_bb_width_5.value) &
                (dataframe['cti'] < self.buy_V_cti_5.value) &
                (dataframe['r_14'] < self.buy_V_r14_5.value) &
                (dataframe['mfi'] < self.buy_V_mfi_5.value) &
                # Really Bear, don't engage until dump over
                (dataframe['ema_vwap_diff_50'] > 0.215) &
                (dataframe['EWO'] < -10)
            )

        is_additional_check = (
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60) &
                (dataframe['volume'] > 0)
            )

        # condition append

        ## Non EWO
        conditions.append(is_cofi)                                                   # ~3.21 90.8%
        dataframe.loc[is_cofi, 'buy_tag'] += 'cofi '

        # EWO > 8
        conditions.append(is_vwap)                                                   # ~67.3%
        dataframe.loc[is_vwap, 'buy_tag'] += 'vwap '

        conditions.append(is_V)                                                      # ~67.9%
        dataframe.loc[is_V, 'buy_tag'] += 'V '

        # EWO 4 ~ 8
        conditions.append(is_lambo_2)                                                # ~67.7%
        dataframe.loc[is_lambo_2, 'buy_tag'] += 'lambo_2 '

        conditions.append(is_clucHA_2)                                               # ~68.2%
        dataframe.loc[is_clucHA_2, 'buy_tag'] += 'cluc_2 '

        conditions.append(is_vwap_2)                                                 # ~67.3%
        dataframe.loc[is_vwap_2, 'buy_tag'] += 'vwap_2 '

        conditions.append(is_V_2)                                                    # ~67.9%
        dataframe.loc[is_V_2, 'buy_tag'] += 'V_2 '

        # EWO -2.5 ~ 4
        conditions.append(is_clucHA_3)                                               # ~68.2%
        dataframe.loc[is_clucHA_3, 'buy_tag'] += 'cluc_3 '

        conditions.append(is_vwap_3)                                                 # ~67.3%
        dataframe.loc[is_vwap_3, 'buy_tag'] += 'vwap_3 '

        # EWO -8 ~ -4
        conditions.append(is_clucHA_4)                                               # ~68.2%
        dataframe.loc[is_clucHA_4, 'buy_tag'] += 'cluc_4 '

        conditions.append(is_vwap_4)                                                 # ~67.3%
        dataframe.loc[is_vwap_4, 'buy_tag'] += 'vwap_4 '

        ## EWO < -8
        conditions.append(is_gumbo)                                                  # ~2.63 / 90.6% / 41.49%      F   (263 %)
        dataframe.loc[is_gumbo, 'buy_tag'] += 'gumbo '

        conditions.append(is_V_5)                                                    # ~67.9%
        dataframe.loc[is_V_5, 'buy_tag'] += 'V_5 '

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
                (dataframe['close'] > dataframe['sma_9']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])

            )
            |
            (
                (dataframe['sma_9'] > (dataframe['sma_9'].shift(1) + dataframe['sma_9'].shift(1) * 0.005 )) &
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
            ] = 1

        return dataframe

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
