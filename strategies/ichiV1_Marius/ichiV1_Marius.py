# --- Do not remove these libs ---
from sqlalchemy import true
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
import copy
import logging
import pathlib
import rapidjson
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_prev_date
from functools import reduce
from datetime import datetime, timedelta, timezone
import numpy as np
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import informative
from freqtrade.strategy import stoploss_from_open
from freqtrade.strategy import (BooleanParameter,timeframe_to_minutes, merge_informative_pair,
                                DecimalParameter, IntParameter, CategoricalParameter)
from freqtrade.persistence import Trade
from typing import Dict
import numpy # noqa
import math
import pandas_ta as pta
from typing import List
from skopt.space import Dimension, Integer
import time
from warnings import simplefilter

from technical.indicators import zema

logger = logging.getLogger(__name__)

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

##### SETINGS #####
# It hyperopt just one set of params for all buy and sell strategies if true.
DUALFIT = False
COUNT = 10
GAP = 3
### END SETINGS ###

def max_pump_detect_price_15m(dataframe, period=14, pause = 288 ):
    df = dataframe.copy()
    df['size'] = df['high'] - df['low']
    cumulativeup = 0
    countup = 0
    cumulativedown = 0
    countdown = 0
    for i in range(period):

        cumulativeup = cumulativeup + df['volume'].shift(i) * df['size'].shift(i) * np.where(df['close'].shift(i) > df['open'].shift(i), 1, 0)
        cumulativedown = cumulativedown + df['volume'].shift(i) * df['size'].shift(i) * np.where(df['close'].shift(i) > df['open'].shift(i), 0, 1)
            
    flow_price = cumulativeup - cumulativedown
    flow_price_normalized = flow_price / (df['volume'].rolling(499).mean() * (df['high']-df['low']).rolling(499).mean())
    max_flow_price = flow_price_normalized.rolling(pause).max()
    
    return max_flow_price

def flow_price_15m(dataframe, period=14, pause = 288 ):
    df = dataframe.copy()
    df['size'] = df['high'] - df['low']
    cumulativeup = 0
    countup = 0
    cumulativedown = 0
    countdown = 0
    for i in range(period):

        cumulativeup = cumulativeup + df['volume'].shift(i) * df['size'].shift(i) * np.where(df['close'].shift(i) > df['open'].shift(i), 1, 0)
        cumulativedown = cumulativedown + df['volume'].shift(i) * df['size'].shift(i) * np.where(df['close'].shift(i) > df['open'].shift(i), 0, 1)
            
    flow_price = cumulativeup - cumulativedown
    flow_price_normalized = flow_price / (df['volume'].rolling(499).mean() * (df['high']-df['low']).rolling(499).mean())
    
    return flow_price_normalized

def to_minutes(**timdelta_kwargs):
    return int(timedelta(**timdelta_kwargs).total_seconds() / 60)

#########################################################
#######################  ichiV1_Mod #####################
#########################################################
      #############################################
class ichiV1_Marius(IStrategy):

    class HyperOpt:
        @staticmethod
        def generate_roi_table(params: dict):
            """
            Generate the ROI table that will be used by Hyperopt
            This implementation generates the default legacy Freqtrade ROI tables.
            Change it if you need different number of steps in the generated
            ROI tables or other structure of the ROI tables.
            Please keep it aligned with parameters in the 'roi' optimization
            hyperspace defined by the roi_space method.
            """
            roi_table = {}
            roi_table[0] = 0.05
            roi_table[params['roi_t6']] = 0.04
            roi_table[params['roi_t5']] = 0.03
            roi_table[params['roi_t4']] = 0.02
            roi_table[params['roi_t3']] = 0.01
            roi_table[params['roi_t2']] = 0.0001
            roi_table[params['roi_t1']] = -10

            return roi_table

        @staticmethod
        def roi_space() -> List[Dimension]:
            """
            Values to search for each ROI steps
            Override it if you need some different ranges for the parameters in the
            'roi' optimization hyperspace.
            Please keep it aligned with the implementation of the
            generate_roi_table method.
            """
            return [
                Integer(240, 720, name='roi_t1'),
                Integer(120, 240, name='roi_t2'),
                Integer(90, 120, name='roi_t3'),
                Integer(60, 90, name='roi_t4'),
                Integer(30, 60, name='roi_t5'),
                Integer(1, 30, name='roi_t6'),
            ]

    DATESTAMP = 0
    SELLMA = 1

    # Buy hyperspace params:
    buy_params = {
        "max_slip": 0.668,
        "antipump_threshold": 0.265,
        "antipump_threshold_2": 0.133,
        "buy_btc_safe_1d": -0.236,
        "buy_btc_safe": -213,
        "buy_threshold": 0.012,
        ##
        "pump_limit": 1000,
        "pump_pause_duration": 192,
        "pump_period": 14,
        "pump_recorver_price": 1.1,
        ##
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "tesla_enabled": True,
       # "buy_fan_magnitude_shift_value": 3,
        "buy_min_fan_magnitude_gain": 1.0022, # NOTE: Good value (Win% ~70%), alot of trades
        #"buy_min_fan_magnitude_gain": 1.008 # NOTE: Very save value (Win% ~90%), only the biggest moves 1.008,
    }

    sell_params = {
        "ProfitLoss1": 0.005,
        "ProfitLoss2": 0.021,
        "ProfitMargin1": 0.018,
        "ProfitMargin2": 0.051,
        "pHSL": -0.08,
        "sell_trend_indicator": "trend_close_2h",
    }

        # minimum conditions to match in buy
    buy_minimum_conditions = IntParameter(
        1, 2, default=1, space="buy", optimize=False, load=True
    )

    position_adjustment_enable = True
    max_dca_orders = 2
    max_dca_multiplier = 1.25
    dca_stake_multiplier = 1.25
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        if (self.config['position_adjustment_enable'] == True) and (self.config['stake_amount'] == 'unlimited'):
            return self.wallets.get_total_stake_amount() / self.config['max_open_trades'] / self.max_dca_multiplier
        else:
            return proposed_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        
        if current_profit > -0.05:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
            return None

        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        if 0 < count_of_buys <= self.max_dca_orders:
            try:
                stake_amount = filled_buys[0].cost
                # This then calculates current safety order size
                stake_amount = stake_amount * self.dca_stake_multiplier
                return stake_amount
            except Exception as exception:
                return None

        return None

    # Pump protection
    pump_period = IntParameter(
        5, 24, default=buy_params['pump_period'], space='buy', optimize=False)
    pump_limit = IntParameter(
        100,10000, default=buy_params['pump_limit'], space='buy', optimize=True)
    pump_recorver_price = DecimalParameter(
        1.0, 1.3, default=buy_params['pump_recorver_price'], space='buy', optimize=True)
    pump_pause_duration = IntParameter(
        6, 500, default=buy_params['pump_pause_duration'], space='buy', optimize=True)

##################################################################    
    ## Slippage params
    is_optimize_slip = False
    max_slip = DecimalParameter(0.33, 0.80, default=0.33, decimals=3, optimize=is_optimize_slip , load=True)
    buy_btc_safe = IntParameter(-300, 50, default=buy_params['buy_btc_safe'], optimize = True)
    buy_btc_safe_1d = DecimalParameter(-0.5, -0.015, default=buy_params['buy_btc_safe_1d'], optimize=True)
    antipump_threshold = DecimalParameter(0, 0.4, default=buy_params['antipump_threshold'], space='buy', optimize=True)
    antipump_threshold_2 = DecimalParameter(0, 0.4, default=buy_params['antipump_threshold_2'], space='buy', optimize=True)

    buy_min_fan_magnitude_gain = DecimalParameter(70, 90, default=buy_params['buy_min_fan_magnitude_gain'], space='buy', optimize=False, load=True)    # Multi Offset
    buy_threshold = DecimalParameter(0.003, 0.012, default=buy_params['buy_threshold'], optimize=True)  

#######################################################################

    # ROI table:
    minimal_roi = {
        "0": 0.215,
        "40": 0.032,
        "87": 0.016,
        "201": 0
    }

    # Stoploss:
    stoploss = -0.275  # value loaded from strategy

    # Trailing stop:
    trailing_stop = False
    #trailing_stop_positive = 0.001
    #trailing_stop_positive_offset = 0.016
    #trailing_only_offset_is_reached = True

    window_buy = IntParameter(60, 1000, default=500, space='buy', optimize=True)
    bandwidth_buy = IntParameter(2, 15, default=8, space='buy', optimize=True)
    mult_buy = DecimalParameter(0.5, 20.0, default=3, space='buy', optimize=True)

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    use_custom_stoploss = True
    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'
    inf_15m = '15m' #use for pump detection
    timeframe_minutes = timeframe_to_minutes(timeframe)
    # storage dict for custom info
    custom_info = {}

    startup_candle_count: int = 499
   # startup_candle_count = 96
    process_only_new_candles = True

    timeperiods = [
        # 50 // timeframe_minutes,
        # 85 // timeframe_minutes,
        180 // timeframe_minutes,
        360 // timeframe_minutes,
        420 // timeframe_minutes,
        560 // timeframe_minutes,
    ]

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # trailing stoploss hyperopt parameters
    pHSL = DecimalParameter(-0.15, -0.08, default=sell_params['pHSL'], decimals=3, space='sell', optimize=True)
    ProfitMargin1 = DecimalParameter(0.009, 0.019, default=sell_params['ProfitMargin1'], decimals=3, space='sell', optimize=True)
    ProfitLoss1 = DecimalParameter(0.005, 0.012, default=sell_params['ProfitLoss1'], decimals=3, space='sell', optimize=True)
    ProfitMargin2 = DecimalParameter(0.033, 0.099, default=sell_params['ProfitMargin2'], decimals=3, space='sell', optimize=True)
    ProfitLoss2 = DecimalParameter(0.010, 0.025, default=sell_params['ProfitLoss2'], decimals=3, space='sell', optimize=True)

    plot_config = {
        'main_plot': {
            # fill area between senkou_a and senkou_b
            'senkou_a': {
                'color': 'green', #optional
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud', #optional
                'fill_color': 'rgba(255,76,46,0.2)', #optional
            },
            # plot senkou_b, too. Not only the area to it.
            'senkou_b': {},

        },
        'subplots': {
            'fan_magnitude': {
                'fan_magnitude': {}
            },
            'fan_magnitude_gain': {
                'fan_magnitude_gain': {}
            }
        }
    }
    
    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        if ((current_time - trade.open_date_utc).seconds / 60 > 1440):
            return 'unclog'

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        HSL = self.pHSL.value
        if (current_profit > self.ProfitMargin2.value):
            sl_profit = self.ProfitLoss2.value
        elif (current_profit > self.ProfitMargin1.value):
            sl_profit = self.ProfitLoss1.value + ((current_profit - self.ProfitMargin1.value) * (self.ProfitLoss2.value - self.ProfitLoss1.value) / (self.ProfitMargin2.value - self.ProfitMargin1.value))
        else:
            sl_profit = HSL

        return sl_profit

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        current_profit = trade.calc_profit_ratio(rate)

        if 'tesla_' in trade.enter_tag and current_profit > 0.01:
            return True

        if (trade.enter_tag == 'telsa_'):
            if (sell_reason in ['exit_signal'])or (sell_reason in ['roi']) or (sell_reason in ['trailing_stop_loss']):
                        return False 

        if (last_candle is not None):
            if (sell_reason in ['exit_signal']):
                if (last_candle['hma_50'] > last_candle['ema_100']) and (last_candle['rsi'] < 45): #*1.2
                    return False

        if (last_candle is not None):
            if (sell_reason in ['exit_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951): #*1.2
                    return False

        # slippage
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True

    age_filter = 30

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['age_filter_ok'] = (dataframe['volume'].rolling(window=self.age_filter, min_periods=self.age_filter).min() > 0)
        return dataframe   

    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs = [(pair, '15m') for pair in pairs]
        informative_pairs.extend([(pair, self.informative_timeframe) for pair in pairs])

        informative_pairs += [("BTC/USDT", "1m")]
        informative_pairs += [("BTC/USDT", "5m")]
        informative_pairs += [("BTC/USDT", "1d")]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe
    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)
        #Weekly average close price  
        informative_1h['weekly_close_avg'] = informative_1h['close'].rolling(168).mean()

        return informative_1h

#######################################################################
    # Informative indicator for pump detection 

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)
        
        informative_15m['max_flow_price'] = max_pump_detect_price_15m(informative_15m, period=self.pump_period.value, pause=self.pump_pause_duration.value)
        informative_15m['flow_price'] = flow_price_15m(informative_15m, period=self.pump_period.value, pause=self.pump_pause_duration.value)
        
        return informative_15m
#######################################################################

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


    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ### BTC protection
        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        btc_df = self.dp.get_pair_dataframe(pair=btc_info_pair, timeframe=self.timeframe)
        dataframe['btc_rsi'] = normalize(ta.RSI(btc_df, timeperiod=14), 0, 100)

        ### BTC protection
        dataframe['btc_5m']= self.dp.get_pair_dataframe('BTC/USDT', timeframe='5m')['close']
        btc_1d = self.dp.get_pair_dataframe('BTC/USDT', timeframe='1d')[['date', 'close']].rename(columns={"close": "btc"}).shift(1)
        dataframe = merge_informative_pair(dataframe, btc_1d, '5m', '1d', ffill=True)

        # BTC info
        informative = self.dp.get_pair_dataframe(pair="BTC/USDT", timeframe="5m")
        informative_past = informative.copy().shift(1)  

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


        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # strategy ClucMay72018
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50) 
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema55'] = ta.EMA(dataframe, timeperiod=55)
        # Pump strength
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']
        dataframe['pump_strength_2'] = (dataframe['ema_50'] - dataframe['ema_200']) / dataframe['ema_50']

        heikinashi = qtpylib.heikinashi(dataframe)
        heikinashi["volume"] = dataframe["volume"]
        dataframe['open'] = heikinashi['open']
        #dataframe['close'] = heikinashi['close']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['close'], timeperiod=72)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['close'], timeperiod=96)

        dataframe['trend_open_5m'] = dataframe['open']
        dataframe['trend_open_15m'] = ta.EMA(dataframe['open'], timeperiod=3)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['open'], timeperiod=6)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['open'], timeperiod=12)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['open'], timeperiod=24)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['open'], timeperiod=48)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['open'], timeperiod=72)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['open'], timeperiod=96)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['chikou_span'] = ichimoku['chikou_span']
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green']
        dataframe['cloud_red'] = ichimoku['cloud_red']

        dataframe['atr'] = ta.ATR(dataframe)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.informative_timeframe, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        #Import 15m indicators    
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)
        drop_columns = [(s + "_" + self.inf_15m) for s in ['date']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)
        
        #Pump protection
        dataframe['weekly_close_avg_offset'] = self.pump_recorver_price.value * dataframe['weekly_close_avg_1h']
        dataframe['price_test'] = dataframe['close'] > dataframe['weekly_close_avg_offset']
        dataframe['pump_price_test'] = dataframe['max_flow_price_15m'] > self.pump_limit.value

        # Check if a pump uccured during pump_pause_duration and coin didn't recovered its pre pump value
        dataframe['pump_dump_alert'] = dataframe['price_test'] & dataframe['pump_price_test']
        dataframe['buy_ok'] = np.where(dataframe['pump_dump_alert'], False, True)
      
        return dataframe

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

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        is_protection = (
            (pct_change(dataframe['btc_1d'], dataframe['btc_5m']).fillna(0) > self.buy_btc_safe_1d.value) &
            (dataframe['pump_strength_2'] < self.antipump_threshold_2.value)&
            (dataframe['buy_ok'])&
            (dataframe['volume'] > 0)
        )

        if self.buy_params['tesla_enabled'] >= True:
            tesla = (

                (dataframe['rsi'] > dataframe['rsi_1h']) &
                (dataframe['trend_close_8h'] > dataframe['trend_close_6h'])&
                (dataframe['trend_close_15m'] > dataframe['trend_close_30m'])&
                (dataframe['trend_open_5m'] > dataframe['trend_open_15m'])&
                (dataframe['trend_close_1h']> dataframe['ema55'])&
                (dataframe['ema21']> dataframe['trend_close_4h'])&
                (dataframe['trend_open_1h'] > dataframe['trend_open_2h'])&
                (dataframe['mfi'] < 70)&
                (dataframe['fan_magnitude_gain'] >= self.buy_min_fan_magnitude_gain.value) &
                (dataframe['fan_magnitude'] > 0.99)
            )
            conditions.append(tesla) 
            dataframe.loc[tesla, 'enter_tag'] += 'tesla_' 

        if conditions:
            dataframe.loc[
                is_protection &
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_params['sell_trend_indicator']]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
        
    def rollingNormalize(self, dataframe, name):
        df = dataframe.copy()
        df[name + '_nmin'] = df[name].rolling(window=1440 // self.timeframe_minutes).min()
        df[name + '_nmax'] = df[name].rolling(window=1440 // self.timeframe_minutes).max()
        return np.where(df[name + '_nmin'] == df[name + '_nmax'], 0, (2.0*(df[name]-df[name + '_nmin'])/(df[name + '_nmax']-df[name + '_nmin'])-1.0))

def normalize(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)


def pct_change(a, b):
    return (b - a) / a

    # smoothed Heiken Ashi
def HA(dataframe, smoothing=None):
    df = dataframe.copy()

    df['HA_Close']=(df['open'] + df['high'] + df['low'] + df['close'])/4

    df.reset_index(inplace=True)

    ha_open = [ (df['open'][0] + df['close'][0]) / 2 ]
    [ ha_open.append((ha_open[i] + df['HA_Close'].values[i]) / 2) for i in range(0, len(df)-1) ]
    df['HA_Open'] = ha_open

    df.set_index('index', inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','high']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','low']].min(axis=1)

    if smoothing is not None:
        sml = abs(int(smoothing))
        if sml > 0:
            df['Smooth_HA_O']=ta.EMA(df['HA_Open'], sml)
            df['Smooth_HA_C']=ta.EMA(df['HA_Close'], sml)
            df['Smooth_HA_H']=ta.EMA(df['HA_High'], sml)
            df['Smooth_HA_L']=ta.EMA(df['HA_Low'], sml)
            
    return df

def pump_warning(dataframe, perc=15):
    df = dataframe.copy()    
    df["change"] = df["high"] - df["low"]
    df["test1"] = (df["close"] > df["open"])
    df["test2"] = ((df["change"]/df["low"]) > (perc/100))
    df["result"] = (df["test1"] & df["test2"]).astype('int')
    return df['result']