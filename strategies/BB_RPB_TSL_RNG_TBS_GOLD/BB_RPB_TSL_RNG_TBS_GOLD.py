# --- Do not remove these libs ---
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta
from typing import Dict, List

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, DatetimeIndex, merge
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from functools import reduce
from technical.indicators import RMI, zema

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
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

class BB_RPB_TSL_RNG_TBS_GOLD(IStrategy):
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
        "buy_fastd": 20,
        "buy_fastk": 22,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 4.179,
        ##
        "buy_ema_high_2": 1.087,
        "buy_ema_low_2": 0.970,
        ##
    }

    # sell space
    sell_params = {
        "pHSL": -0.178,
        "pPF_1": 0.019,
        "pPF_2": 0.065,
        "pSL_1": 0.019,
        "pSL_2": 0.062,
        "sell_btc_safe": -389,
        "base_nb_candles_sell": 24,
        "high_offset": 0.991,
        "high_offset_2": 0.997        
    }

    # really hard to use this
    minimal_roi = {
        "0": 0.10,
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    # Disabled
    stoploss = -0.049

    # Custom stoploss
    use_custom_stoploss = True
    use_sell_signal = True
    process_only_new_candles = True
    ############################################################################

    ## Buy params

    is_optimize_dip = False
    buy_rmi = IntParameter(30, 50, default=35, optimize= is_optimize_dip)
    buy_cci = IntParameter(-135, -90, default=-133, optimize= is_optimize_dip)
    buy_srsi_fk = IntParameter(30, 50, default=25, optimize= is_optimize_dip)
    buy_cci_length = IntParameter(25, 45, default=25, optimize = is_optimize_dip)
    buy_rmi_length = IntParameter(8, 20, default=8, optimize = is_optimize_dip)

    is_optimize_break = False
    buy_bb_width = DecimalParameter(0.05, 0.2, default=0.15, optimize = is_optimize_break)
    buy_bb_delta = DecimalParameter(0.025, 0.08, default=0.04, optimize = is_optimize_break)

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

    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    is_optimize_btc_safe = False
    buy_btc_safe = IntParameter(-300, 50, default=-200, optimize = is_optimize_btc_safe)
    buy_btc_safe_1d = DecimalParameter(-0.075, -0.025, default=-0.05, optimize = is_optimize_btc_safe)
    buy_threshold = DecimalParameter(0.003, 0.012, default=0.008, optimize = is_optimize_btc_safe)

    # Buy params toggle
    buy_is_dip_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_is_break_enabled = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)

    ## Sell params
    sell_btc_safe = IntParameter(-400, -300, default=-365, optimize = True)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    high_offset          = DecimalParameter(0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2        = DecimalParameter(0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)      

    ## Trailing params

    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    ############################################################################

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.timeframe) for pair in pairs]
        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"
        informative_pairs.append((btc_info_pair, self.timeframe))
        #informative_pairs = [("BTC/BUSD", "5m")]

        return informative_pairs

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert self.dp, "DataProvider is required for multiple timeframes."

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
        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"
        inf_tf = '5m'
        informative = self.dp.get_pair_dataframe(btc_info_pair, timeframe=inf_tf)
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
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # EMA
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)   
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)        

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


        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)        

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        if self.buy_is_dip_enabled.value:

            is_dip = (
                (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value)
            )

            #conditions.append(is_dip)

        if self.buy_is_break_enabled.value:

            is_break = (

                (   (dataframe['bb_delta'] > self.buy_bb_delta.value)                                   #"buy_bb_delta": 0.025 0.036
                    &                                                                                   #"buy_bb_width": 0.095 0.133
                    (dataframe['bb_width'] > self.buy_bb_width.value)
                )
                &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &    # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value)
            )
            #conditions.append(is_break)

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

        # is_btc_safe = (

        #         (dataframe['btc_diff'] > self.buy_btc_safe.value)
        #        &(dataframe['btc_5m'] - dataframe['btc_1d'] > dataframe['btc_1d'] * self.buy_btc_safe_1d.value)
        #        &(dataframe['volume'] > 0)           # Make sure Volume is not 0
        #     )

        is_BB_checked = is_dip & is_break

        #print(dataframe['btc_5m'])
        #print(dataframe['btc_1d'])
        #print(dataframe['btc_5m'] - dataframe['btc_1d'])
        #print(dataframe['btc_1d'] * -0.025)
        #print(dataframe['btc_5m'] - dataframe['btc_1d'] > dataframe['btc_1d'] * -0.025)

        ## condition append
        conditions.append(is_BB_checked)          # ~1.7 89%
        dataframe.loc[is_BB_checked, 'buy_tag'] += 'bb '

        conditions.append(is_local_uptrend)       # ~3.84 90.2%
        dataframe.loc[is_local_uptrend, 'buy_tag'] += 'local uptrend '

        conditions.append(is_ewo)                 # ~2.26 93.5%
        dataframe.loc[is_ewo, 'buy_tag'] += 'ewo '

        conditions.append(is_ewo_2)               # ~3.68 90.3%
        dataframe.loc[is_ewo_2, 'buy_tag'] += 'ewo2 '

        conditions.append(is_cofi)                # ~3.21 90.8%
        dataframe.loc[is_cofi, 'buy_tag'] += 'cofi '

        conditions.append(is_nfi_32)              # ~2.43 91.3%
        dataframe.loc[is_nfi_32, 'buy_tag'] += 'nfi 32 '

        conditions.append(is_nfi_33)              # ~0.11 100%
        dataframe.loc[is_nfi_33, 'buy_tag'] += 'nfi 33 '

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'buy' ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (                   
                (dataframe['close'] > dataframe['sma_9'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi']>50)&
                (dataframe['volume'] > 0)&                
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])

            )
            |
            (
                (dataframe['sma_9'] > (dataframe['sma_9'].shift(1) + dataframe['sma_9'].shift(1)*0.005 )) &
                (dataframe['close'] < dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']>dataframe['rsi_slow'])
            )    
            
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
        
        
class TrailingBuyStrat2(BB_RPB_TSL_RNG_TBS_GOLD):
    # Original idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal and will move to next candles if the trailing still active
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #

    process_only_new_candles = True

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
            logger.info(
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
                            logger.info(f'start trailing buy for {pair} at {last_candle["close"]}')

                        elif trailing_buy['trailing_buy_order_started']:
                            if trailing_buy_offset == 'forcebuy':
                                # buy in custom conditions
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

                            elif trailing_buy_offset is None:
                                # stop trailing buy custom conditions
                                self.trailing_buy(pair, reinit=True)
                                logger.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')

                            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                # update uplimit
                                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                            elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                                # buy ! current price > uplimit && lower thant starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price price ({(trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy))}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")

                            elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                                # stop trailing buy because price is too high
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'price too high for {pair} !')

                    else:
                        logger.info(f"Wait for next buy signal for {pair}")

                if (val == True):
                    self.trailing_buy_info(pair, rate)
                    self.trailing_buy(pair, reinit=True)
                    logger.info(f'STOP trailing buy for {pair} because I buy it')
        
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
                        logger.info(f"Set 'allow_trailing' to True for {metadata['pair']} to start trailing!!!")
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_buy['allow_trailing'] = True
                        initial_buy_tag = last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal'
                        dataframe.loc[:, 'buy_tag'] = f"{initial_buy_tag} (start trail price {last_candle['close']})"
            else:
                if (trailing_buy['trailing_buy_order_started'] == True):
                    logger.info(f"Continue trailing for {metadata['pair']}. Manually trigger buy signal!!")
                    dataframe.loc[:,'buy'] = 1
                    dataframe.loc[:, 'buy_tag'] = trailing_buy['buy_tag']
                    # dataframe['buy'] = 1

        return dataframe

        
