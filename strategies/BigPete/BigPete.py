import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from typing import Dict, List
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from functools import reduce


###########################################################################################################
##                   BigZ04_TSL4 by Perkmeister, based on BigZ04 by ilya                                 ##
##                                                                                                       ##
##    https://github.com/i1ya/freqtrade-strategies                                                       ##
##    The stratagy most inspired by iterativ (authors of the CombinedBinHAndClucV6)                      ##
##                                                                                                       ##
##    This is a modified version of BigZ04 that uses custom_stoploss() to implement a hard stoploss      ##
##    of 8%, and to replace the roi table with a trailing stoploss to extract more profit when prices    ##
##    start to rise above a profit threshold. It's quite simple and crude and is a 'first stab' at the   ##
##    hard stoploss problem, use live at your own risk ;). The sell signals from SMAOffsetProtectOptV1   ##
##    have been added but are currently disabled as had no benefit.                                      ##
##                                                                                                       ##
###########################################################################################################
##     The main point of this strat is:                                                                  ##
##        -  make drawdown as low as possible                                                            ##
##        -  buy at dip                                                                                  ##
##        -  soft check if market if rising                                                              ##
##        -  hard check is market if fallen                                                              ##
##        -  11 buy signals                                                                              ##
##        -  hard stoploss function preventing from big fall                                             ##
##        -  trailing stoploss while in profit                                                           ##
##        -  no sell signal. Uses custom stoploss                                                        ##
##                                                                                                       ##
###########################################################################################################
##                 GENERAL RECOMMENDATIONS                                                               ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 2 and 4 open trades, with unlimited stake.        ##
##                                                                                                       ##
##   As a pairlist it is recommended to use a static pairlst such as iterativ's orginal:                 ##
##   https://discord.com/channels/700048804539400213/702584639063064586/838038600368783411               ## 
##                                                                                                       ##
##   Ensure that you don't override any variables in your config.json. Especially                        ##
##   the timeframe (must be 5m).                                                                         ##
##                                                                                                       ##
##                                                                                                       ##
###########################################################################################################
##               DONATIONS 2 @iterativ (author of the original strategy)                                 ##
##                                                                                                       ##
##   Absolutely not required. However, will be accepted as a token of appreciation.                      ##
##                                                                                                       ##
##   BTC: bc1qvflsvddkmxh7eqhc4jyu5z5k6xcw3ay8jl49sk                                                     ##
##   ETH: 0x83D3cFb8001BDC5d2211cBeBB8cB3461E5f7Ec91                                                     ##
##                                                                                                       ##
###########################################################################################################

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


class BigPete(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 100.0
    }
    
    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'
    inf_1h = '1h'

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = False

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.0187
    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    optimize_cond = True
    opzimize_buy_params = False
    
    buy_params = {
      "buy_bb20_close_bblowerband_safe_1": 0.999,
      "buy_bb20_close_bblowerband_safe_2": 1.01,
      "buy_condition_0_enable": True,
      "buy_condition_10_enable": True,
      "buy_condition_11_enable": True,
      "buy_condition_12_enable": True,
      "buy_condition_13_enable": False,
      "buy_condition_1_enable": True,
      "buy_condition_2_enable": True,
      "buy_condition_3_enable": True,
      "buy_condition_4_enable": True,
      "buy_condition_5_enable": True,
      "buy_condition_6_enable": True,
      "buy_condition_7_enable": True,
      "buy_condition_8_enable": True,
      "buy_condition_9_enable": True,
      "buy_dip_0": 1.029,
      "buy_macd_1": 0.05,
      "buy_macd_2": 0.03,
      "buy_rsi_0": 11.2,
      "buy_rsi_1": 15.7,
      "buy_rsi_1h_0": 81.7,
      "buy_rsi_1h_1": 14.2,
      "buy_rsi_1h_1a": 67.8,
      "buy_rsi_1h_2": 23.5,
      "buy_rsi_1h_3": 21.6,
      "buy_rsi_1h_4": 31.5,
      "buy_rsi_1h_5": 31.3,
      "buy_rsi_2": 11.3,
      "buy_rsi_3": 35.6,
      "buy_volume_drop_1": 5.4,
      "buy_volume_drop_2": 1.6,
      "buy_volume_drop_3": 9.7,
      "buy_volume_pump_1": 0.1
    }
#   Optimize custom Stop-Loss, trailing + profit threshold
    opzimize_cSL=True 		#default True
    optimize_trailing=True 	#default True
    optimize_SLT=False		#default False
    # V1 original
    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 49,
        "high_offset": 1.006,
        "narrow_stop": 1.02,
        "pHSL": -0.08,
        "pPF_1": 0.016,
        "pPF_2": 0.08,
        "pSL_1": 0.011,
        "pSL_2": 0.04,
        "rsi_1h_val": 32,
        "trade_time": 35,
        "wide_stop": 1.035

    }
    
    
    ############################################################################

    # Buy

    buy_condition_0_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_0_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_1_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_1_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_2_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_2_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_3_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_3_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_4_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_4_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_5_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_5_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_6_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_6_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_7_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_7_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_8_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_8_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_9_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_9_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_10_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_10_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_11_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_11_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_12_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_12_enable'], space='buy', optimize=optimize_cond, load=True)
    buy_condition_13_enable = CategoricalParameter([True, False], default=buy_params['buy_condition_13_enable'], space='buy', optimize=optimize_cond, load=True)

    buy_bb20_close_bblowerband_safe_1 = DecimalParameter(0.950, 1.050, default=buy_params['buy_bb20_close_bblowerband_safe_1'], decimals=3, space='buy', optimize=opzimize_buy_params, load=True)
    buy_bb20_close_bblowerband_safe_2 = DecimalParameter(0.700, 1.100, default=buy_params['buy_bb20_close_bblowerband_safe_2'], decimals=2, space='buy', optimize=opzimize_buy_params, load=True)

    buy_volume_pump_1 = DecimalParameter(0.1, 0.9, default=buy_params['buy_volume_pump_1'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_volume_drop_1 = DecimalParameter(1, 10, default=buy_params['buy_volume_drop_1'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_volume_drop_2 = DecimalParameter(1, 10, default=buy_params['buy_volume_drop_2'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_volume_drop_3 = DecimalParameter(1, 10, default=buy_params['buy_volume_drop_3'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)

    buy_rsi_1h_0 = DecimalParameter(55.0, 85.0, default=buy_params['buy_rsi_1h_0'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)    
    buy_rsi_1h_1a = DecimalParameter(65.0, 78.0, default=buy_params['buy_rsi_1h_1a'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_rsi_1h_1 = DecimalParameter(10.0, 40.0, default=buy_params['buy_rsi_1h_1'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_rsi_1h_2 = DecimalParameter(10.0, 40.0, default=buy_params['buy_rsi_1h_2'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_rsi_1h_3 = DecimalParameter(10.0, 40.0, default=buy_params['buy_rsi_1h_3'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_rsi_1h_4 = DecimalParameter(10.0, 40.0, default=buy_params['buy_rsi_1h_4'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_rsi_1h_5 = DecimalParameter(10.0, 60.0, default=buy_params['buy_rsi_1h_5'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)

    buy_rsi_0 = DecimalParameter(10.0, 40.0, default=buy_params['buy_rsi_0'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_rsi_1 = DecimalParameter(10.0, 40.0, default=buy_params['buy_rsi_1'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_rsi_2 = DecimalParameter(7.0, 40.0, default=buy_params['buy_rsi_2'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)
    buy_rsi_3 = DecimalParameter(7.0, 40.0, default=buy_params['buy_rsi_3'], space='buy', decimals=1, optimize=opzimize_buy_params, load=True)

    buy_macd_1 = DecimalParameter(0.01, 0.09, default=buy_params['buy_macd_1'], space='buy', decimals=2, optimize=opzimize_buy_params, load=True)
    buy_macd_2 = DecimalParameter(0.01, 0.09, default=buy_params['buy_macd_2'], space='buy', decimals=2, optimize=opzimize_buy_params, load=True)

    buy_dip_0 = DecimalParameter(1.015, 1.040, default=buy_params['buy_dip_0'], space='buy', decimals=3, optimize=opzimize_buy_params, load=True)


    # hyperopt parameters for custom_stoploss()
    trade_time = IntParameter(25, 65, default=sell_params['trade_time'], space='sell', optimize=opzimize_cSL, load=True)
    rsi_1h_val = IntParameter(25, 45, default=sell_params['rsi_1h_val'], space='sell', optimize=opzimize_cSL, load=True)
    narrow_stop = DecimalParameter(1.005, 1.030, default=sell_params['narrow_stop'], space='sell', decimals=3, optimize=opzimize_cSL, load=True)
    wide_stop = DecimalParameter(1.010, 1.045, default=sell_params['wide_stop'], space='sell', decimals=3, optimize=opzimize_cSL, load=True)

    # hyperopt parameters for SMAOffsetProtectOptV1 sell signal
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False, load=True)
    high_offset = DecimalParameter(0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=False, load=True)

    # trailing stoploss hyperopt parameters
    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=sell_params['pHSL'], decimals=3, space='sell', optimize=optimize_trailing, load=True)

    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=sell_params['pPF_1'], decimals=3, space='sell', optimize=optimize_SLT, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=sell_params['pSL_1'], decimals=3, space='sell', optimize=optimize_SLT, load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=sell_params['pPF_2'], decimals=3, space='sell', optimize=optimize_SLT, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=sell_params['pSL_2'], decimals=3, space='sell', optimize=optimize_SLT, load=True)


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        return True


    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        return False


    # new custom stoploss, both hard and trailing functions. Trailing stoploss first rises at a slower
    # rate than the current rate until a profit threshold is reached, after which it rises at a constant
    # percentage as per a normal trailing stoploss. This allows more margin for pull-backs during a rise.
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
            sl_profit = SL_1 + ((current_profit - PF_1)*(SL_2 - SL_1)/(PF_2 - PF_1))
        else:
            sl_profit = HSL
        
        return stoploss_from_open(sl_profit, current_profit)
    
        
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        informative_1h['ema_50'] = ta.SMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.SMA(informative_1h, timeperiod=200)
        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband'] = bollinger['lower']
        informative_1h['bb_middleband'] = bollinger['mid']
        informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=48).mean()

        # EMA
        dataframe['ema_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        # MACD 
        dataframe['macd'], dataframe['signal'], dataframe['hist'] = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # SMA
        dataframe['sma_5'] = ta.EMA(dataframe, timeperiod=5)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ------ ATR stuff
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

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

        cond13 = (self.buy_condition_12_enable.value & (dataframe['close'] > dataframe['ema_200']) & (dataframe['close'] > dataframe['ema_200_1h']) & 
                            (dataframe['close'] < dataframe['bb_lowerband'] * 0.993) &
                            (dataframe['low'] < dataframe['bb_lowerband'] * 0.985) &
                            (dataframe['close'].shift() > dataframe['bb_lowerband']) &
                            (dataframe['rsi_1h'] < 72.8) &
                            (dataframe['open'] > dataframe['close']) &
                            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                            ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &
                            (dataframe['volume'] > 0).astype(int))
        

        cond11 = (self.buy_condition_11_enable.value &
                            (dataframe['close'] > dataframe['ema_200']) &
                            (dataframe['hist'] > 0) &
                            (dataframe['hist'].shift() > 0) &
                            (dataframe['hist'].shift(2) > 0) &
                            (dataframe['hist'].shift(3) > 0) &
                            (dataframe['hist'].shift(5) > 0) &
                            (dataframe['bb_middleband'] - dataframe['bb_middleband'].shift(5) > dataframe['close']/200) &
                            (dataframe['bb_middleband'] - dataframe['bb_middleband'].shift(10) > dataframe['close']/100) &
                            ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) < (dataframe['close']*0.1)) &
                            ((dataframe['open'].shift() - dataframe['close'].shift()) < (dataframe['close'] * 0.018)) &
                            (dataframe['rsi'] > 51) &
                            (dataframe['open'] < dataframe['close']) &
                            (dataframe['open'].shift() > dataframe['close'].shift()) &
                            (dataframe['close'] > dataframe['bb_middleband']) &
                            (dataframe['close'].shift() < dataframe['bb_middleband'].shift()) &
                            (dataframe['low'].shift(2) > dataframe['bb_middleband'].shift(2)) &
                            (dataframe['volume'] > 0)).astype(int)  # Make sure Volume is not 0
        

        cond0 = (self.buy_condition_0_enable.value &
                            (dataframe['close'] > dataframe['ema_200']) &
                            (dataframe['rsi'] < self.buy_rsi_0.value) &
                            ((dataframe['close'] * self.buy_dip_0.value < dataframe['open'].shift(3)) | 
                            (dataframe['close'] * self.buy_dip_0.value< dataframe['open'].shift(2)) |
                            (dataframe['close'] * self.buy_dip_0.value < dataframe['open'].shift(1))) &
                            (dataframe['rsi_1h'] < self.buy_rsi_1h_0.value) &
                            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                            (dataframe['volume'] > 0)).astype(int)  # Make sure Volume is not 0
            
      
        cond1 = (self.buy_condition_1_enable.value &

                            (dataframe['close'] > dataframe['ema_200']) &
                            (dataframe['close'] > dataframe['ema_200_1h']) &

                            (dataframe['close'] <  dataframe['bb_lowerband'] * self.buy_bb20_close_bblowerband_safe_1.value) &
                            (dataframe['rsi_1h'] < self.buy_rsi_1h_1a.value) &
                            (dataframe['open'] > dataframe['close']) &
                            
                            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                            ((dataframe['open'] - dataframe['close']) < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &

                            (dataframe['volume'] > 0)).astype(int) 
                    

        cond2 = (self.buy_condition_2_enable.value &(dataframe['close'] > dataframe['ema_200']) &(dataframe['close'] < dataframe['bb_lowerband'] *  self.buy_bb20_close_bblowerband_safe_2.value) &(dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &(dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &(dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &(dataframe['open'] - dataframe['close'] < dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) &(dataframe['volume'] > 0))
        

        cond3 = (self.buy_condition_3_enable.value &
                            (dataframe['close'] > dataframe['ema_200_1h']) &
                            (dataframe['close'] < dataframe['bb_lowerband']) &
                            (dataframe['rsi'] < self.buy_rsi_3.value) &
                            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_3.value)) &
                            (dataframe['volume'] > 0)).astype(int) 
        

        cond4 = (self.buy_condition_4_enable.value &
                            (dataframe['rsi_1h'] < self.buy_rsi_1h_1.value) &
                            (dataframe['close'] < dataframe['bb_lowerband']) &
                            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                            (dataframe['volume'] > 0)).astype(int) 
        

        cond5 = (self.buy_condition_5_enable.value &

                            (dataframe['close'] > dataframe['ema_200']) &
                            (dataframe['close'] > dataframe['ema_200_1h']) &

                            (dataframe['ema_26'] > dataframe['ema_12']) &
                            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
                            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                            (dataframe['close'] < (dataframe['bb_lowerband'])) &

                            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                            (dataframe['volume'] > 0)).astype(int) 
            

        cond6 = (self.buy_condition_6_enable.value &

                            (dataframe['rsi_1h'] < self.buy_rsi_1h_5.value) &

                            (dataframe['ema_26'] > dataframe['ema_12']) &
                            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_2.value)) &
                            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                            (dataframe['close'] < (dataframe['bb_lowerband'])) &

                            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                            (dataframe['volume'] > 0)).astype(int) 
            

        cond7 = (self.buy_condition_7_enable.value &

                            (dataframe['rsi_1h'] < self.buy_rsi_1h_2.value) &
                            
                            (dataframe['ema_26'] > dataframe['ema_12']) &
                            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_macd_1.value)) &
                            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                            
                            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                            (dataframe['volume'] > 0)).astype(int) 


        cond8 = (self.buy_condition_8_enable.value &

                            (dataframe['rsi_1h'] < self.buy_rsi_1h_3.value) &
                            (dataframe['rsi'] < self.buy_rsi_1.value) &
                            
                            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &

                            (dataframe['volume'] > 0)).astype(int) 

        cond9 = (self.buy_condition_9_enable.value &

                            (dataframe['rsi_1h'] < self.buy_rsi_1h_4.value) &
                            (dataframe['rsi'] < self.buy_rsi_2.value) &
                            
                            (dataframe['volume'] < (dataframe['volume'].shift() * self.buy_volume_drop_1.value)) &
                            (dataframe['volume_mean_slow'] > dataframe['volume_mean_slow'].shift(48) * self.buy_volume_pump_1.value) &
                            (dataframe['volume_mean_slow'] * self.buy_volume_pump_1.value < dataframe['volume_mean_slow'].shift(48)) &
                            (dataframe['volume'] > 0)).astype(int) 

        cond10 = (self.buy_condition_10_enable.value &

                            (dataframe['rsi_1h'] < self.buy_rsi_1h_4.value) &
                            (dataframe['close_1h'] < dataframe['bb_lowerband_1h']) &

                            (dataframe['hist'] > 0) &
                            (dataframe['hist'].shift(2) < 0) &
                            (dataframe['rsi'] < 40.5) &
                            (dataframe['hist'] > dataframe['close'] * 0.0012) &
                            (dataframe['open'] < dataframe['close']) &
                            (dataframe['volume'] > 0)).astype(int) 

        dataframe.loc[((cond1 + cond2 + cond3 + cond4 + cond5 + cond6 + cond7 + cond8 + cond9 + cond10 + cond11 + cond13) >= 1), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middleband'] * 1.01) &                  # Don't be gready, sell fast
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            ,
            'sell'
        ] = 0

        return dataframe