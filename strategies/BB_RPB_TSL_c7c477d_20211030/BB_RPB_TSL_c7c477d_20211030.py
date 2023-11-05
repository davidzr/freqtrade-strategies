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
from technical.indicators import RMI, zema, ichimoku

# --------------------------------
def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

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
        fillna(bool): if True, fill nan values.
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

class BB_RPB_TSL_c7c477d_20211030(IStrategy):
    '''
        BB_RPB_TSL
        @author jilv220
        Simple bollinger brand strategy inspired by this blog  ( https://hacks-for-life.blogspot.com/2020/12/freqtrade-notes.html )
        RPB, which stands for Real Pull Back, taken from ( https://github.com/GeorgeMurAlkh/freqtrade-stuff/blob/main/user_data/strategies/TheRealPullbackV2.py )
        The trailing custom stoploss taken from BigZ04_TSL from Perkmeister ( modded by ilya )
        I modified it to better suit my taste and added Hyperopt for this strategy.
    '''

    ##########################################################################

    # Hyperopt result area

    # buy space
    buy_params = {
        ##
        "buy_btc_safe": -289,
        "buy_btc_safe_1d": -0.05,
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
        "buy_closedelta": 12.148,
        "buy_ema_diff": 0.022,
        ##
        "buy_adx": 20,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 2.055,
        "buy_fastd": 21,
        "buy_fastk": 30,
        ##
        "buy_ema_high_2": 1.087,
        "buy_ema_low_2": 0.970,
    }

    # sell space
    sell_params = {
        ##
        "pHSL": -0.178,
        "pPF_1": 0.01,
        "pPF_2": 0.048,
        "pSL_1": 0.009,
        "pSL_2": 0.043,
        ##
        "sell_btc_safe": -389,
        "sell_cmf": -0.046,
        "sell_ema": 0.988,
        "sell_ema_close_delta": 0.022,
    }

    # really hard to use this
    minimal_roi = {
        "0": 0.10,
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    # Disabled
    stoploss = -0.99

    # Custom stoploss
    use_custom_stoploss = True
    use_sell_signal = True

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

    is_optimize_local_dip = False
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_dip)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize = False)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_dip)

    is_optimize_ewo = False
    buy_rsi_fast = IntParameter(35, 50, default=45, optimize = False)
    buy_rsi = IntParameter(15, 30, default=35, optimize = False)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.585, optimize = is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942 , optimize = is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.084 , optimize = is_optimize_ewo)

    is_optimize_ewo_2 = False
    buy_ema_low_2 = DecimalParameter(0.96, 0.978, default=0.96 , optimize = is_optimize_ewo_2)
    buy_ema_high_2 = DecimalParameter(1.05, 1.2, default=1.09 , optimize = is_optimize_ewo_2)

    is_optimize_cofi = True
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = False)

    is_optimize_btc_safe = False
    buy_btc_safe = IntParameter(-300, 50, default=-200, optimize = is_optimize_btc_safe)
    buy_btc_safe_1d = DecimalParameter(-0.075, -0.025, default=-0.05, optimize = is_optimize_btc_safe)
    buy_threshold = DecimalParameter(0.003, 0.012, default=0.008, optimize = is_optimize_btc_safe)

    ## Sell params

    sell_btc_safe = IntParameter(-400, -300, default=-365, optimize = False)

    is_optimize_sell_stoploss = False
    sell_cmf = DecimalParameter(-0.4, 0.0, default=0.0, optimize = is_optimize_sell_stoploss)
    sell_ema_close_delta = DecimalParameter(0.022, 0.027, default= 0.024, optimize = is_optimize_sell_stoploss)
    sell_ema = DecimalParameter(0.97, 0.99, default=0.987 , optimize = is_optimize_sell_stoploss)

    ## Trailing params

    is_optimize_trailing = True
    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', optimize=is_optimize_trailing , load=True)

    ############################################################################

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs += [("BTC/USDT", "5m")]

        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        # Ichimoku
        ichi = ichimoku(informative_1h, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        informative_1h['chikou_span'] = ichi['chikou_span']
        informative_1h['tenkan_sen'] = ichi['tenkan_sen']
        informative_1h['kijun_sen'] = ichi['kijun_sen']
        informative_1h['senkou_a'] = ichi['senkou_span_a']
        informative_1h['senkou_b'] = ichi['senkou_span_b']
        informative_1h['leading_senkou_span_a'] = ichi['leading_senkou_span_a']
        informative_1h['leading_senkou_span_b'] = ichi['leading_senkou_span_b']
        informative_1h['chikou_span_greater'] = (informative_1h['chikou_span'] > informative_1h['senkou_a']).shift(30).fillna(False)
        informative_1h.loc[:, 'cloud_top'] = informative_1h.loc[:, ['senkou_a', 'senkou_b']].max(axis=1)

        return informative_1h

    ############################################################################

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

    ############################################################################

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Bollinger bands (hyperopt hard to implement)
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        ### BTC protection

        # BTC info
        inf_tf = '5m'
        informative = self.dp.get_pair_dataframe('BTC/USDT', timeframe=inf_tf)
        informative_past = informative.copy().shift(1)                                                                                                   # Get recent BTC info

        # BTC 5m dump protection
        informative_past_source = (informative_past['open'] + informative_past['close'] + informative_past['high'] + informative_past['low']) / 4        # Get BTC price
        informative_threshold = informative_past_source * self.buy_threshold.value                                                                       # BTC dump n% in 5 min
        informative_past_delta = informative_past['close'].shift(1) - informative_past['close']                                                          # should be positive if dump
        informative_diff = informative_threshold - informative_past_delta                                                                                # Need be larger than 0
        dataframe['btc_threshold'] = informative_threshold
        dataframe['btc_diff'] = informative_diff

        # BTC 1d dump protection
        informative_past_1d = informative.copy().shift(288)
        informative_past_source_1d = (informative_past_1d['open'] + informative_past_1d['close'] + informative_past_1d['high'] + informative_past_1d['low']) / 4
        dataframe['btc_5m'] = informative_past_source
        dataframe['btc_1d'] = informative_past_source_1d

        ### Other checks

        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])
        dataframe['bb_bottom_cross'] = qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband3']).astype('int')

        # CCI hyperopt
        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cci'] = ta.CCI(dataframe, 26)
        dataframe['cci_long'] = ta.CCI(dataframe, 170)

        # RMI hyperopt
        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)
        #dataframe['rmi'] = RMI(dataframe, length=8, mom=4)

        # SRSI hyperopt ?
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        # BinH
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # SMA
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # EMA
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # Elliot
        dataframe['EWO'] = EWO(dataframe, 50, 200)

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        # Volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)

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
        dataframe.loc[:, 'buy_tag'] = ''

        is_dip = (
                (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value)
            )

        is_break = (

                (   (dataframe['bb_delta'] > self.buy_bb_delta.value)                                   #"buy_bb_delta": 0.025 0.036
                    &                                                                                   #"buy_bb_width": 0.095 0.133
                    (dataframe['bb_width'] > self.buy_bb_width.value)
                )
                &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &    # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value)
            )

        is_ichi_ok = (

                (dataframe['tenkan_sen_1h'] > dataframe['kijun_sen_1h']) &
                (dataframe['close'] > dataframe['cloud_top_1h']) &
                (dataframe['leading_senkou_span_a_1h'] > dataframe['leading_senkou_span_b_1h']) &
                (dataframe['chikou_span_greater_1h'])
        )

        is_local_uptrend = (                                                                            # from NFI next gen

                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 )
            )

        is_ewo = (                                                                                      # from SMA offset
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                (dataframe['EWO'] > self.buy_ewo.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
            )

        is_ewo_2 = (
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low_2.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high_2.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
            )

        is_cofi = (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value)
            )

        # NFI quick mode

        is_nfi_32 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 46) &
                (dataframe['rsi'] > 19) &
                (dataframe['close'] < dataframe['sma_15'] * 0.942) &
                (dataframe['cti'] < -0.86)
            )

        is_nfi_33 = (
                (dataframe['close'] < (dataframe['ema_13'] * 0.978)) &
                (dataframe['EWO'] > 8) &
                (dataframe['cti'] < -0.88) &
                (dataframe['rsi'] < 32) &
                (dataframe['r_14'] < -98.0) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.5))
            )

        is_btc_safe = (

                (dataframe['btc_diff'] > self.buy_btc_safe.value)
               &(dataframe['btc_5m'] - dataframe['btc_1d'] > dataframe['btc_1d'] * self.buy_btc_safe_1d.value)
               &(dataframe['volume'] > 0)           # Make sure Volume is not 0
            )

        is_BB_checked = is_dip & is_break
        is_cofi_checked = is_cofi & is_ichi_ok

        #print(dataframe['btc_5m'])
        #print(dataframe['btc_1d'])
        #print(dataframe['btc_5m'] - dataframe['btc_1d'])
        #print(dataframe['btc_1d'] * -0.025)
        #print(dataframe['btc_5m'] - dataframe['btc_1d'] > dataframe['btc_1d'] * -0.025)

        # condition append
        conditions.append(is_BB_checked)                                           # ~1.61 / 87.9% / 29.36%
        dataframe.loc[is_BB_checked, 'buy_tag'] += 'bb '

        conditions.append(is_local_uptrend)                                        # ~3.5 / 89.9% / 56.4%
        dataframe.loc[is_local_uptrend, 'buy_tag'] += 'local uptrend '

        conditions.append(is_ewo)                                                  # ~2.19 / 92.6% / 28.12%
        dataframe.loc[is_ewo, 'buy_tag'] += 'ewo '

        conditions.append(is_ewo_2)                                                # ~3.65 / 82.5% / 21.56%
        dataframe.loc[is_ewo_2, 'buy_tag'] += 'ewo2 '

        conditions.append(is_cofi_checked)                                         # ~2.57 / 82.2% / 52.42%
        dataframe.loc[is_cofi_checked, 'buy_tag'] += 'cofi '

        conditions.append(is_nfi_32)                                               # ~2.3 / 88.2% / 42.35%
        dataframe.loc[is_nfi_32, 'buy_tag'] += 'nfi 32 '

        conditions.append(is_nfi_33)                                               # ~0.11 / 100%
        dataframe.loc[is_nfi_33, 'buy_tag'] += 'nfi 33 '

        if conditions:
            dataframe.loc[ is_btc_safe & reduce(lambda x, y: x | y, conditions), 'buy' ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (
                    (dataframe['btc_diff'] < self.sell_btc_safe.value)
                    |
                    (
                        (dataframe['close'] < dataframe['ema_200'] * self.sell_ema.value) &
                        (dataframe['cmf'] < self.sell_cmf.value) &
                        (((dataframe['ema_200'] - dataframe['close']) / dataframe['close']) < self.sell_ema_close_delta.value) &
                        (dataframe['rsi'] > dataframe['rsi'].shift(1))
                    )
                )
                &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
