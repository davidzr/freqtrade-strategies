# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import math

"""
    https://fr.tradingview.com/script/vDX9m7PJ-L2-KDJ-with-Whale-Pump-Detector/
    translated for freqtrade: viksal1982  viktors.s@gmail.com
"""


def xsa(dataframe, source, len, wei):
    df = dataframe.copy().fillna(0)
    def calc_xsa(dfr, init=0):
        global calc_sumf_value
        global calc_src_value
        global calc_out_value
        if init == 1:
            calc_sumf_value = [0.0] * len
            calc_src_value = [0.0] * len
            calc_out_value = [0.0] * len
            return
        calc_src_value.pop(0)
        calc_src_value.append(dfr[source])
        sumf_val = calc_sumf_value[-1] - calc_src_value[0] 
        ma_val = sumf_val / len
        out_val = (calc_src_value[-1] * wei + calc_out_value[-1] * (len-wei))/len
        calc_sumf_value.pop(0)
        calc_sumf_value.append(sumf_val)
        calc_out_value.pop(0)
        calc_out_value.append(out_val)
        return out_val
    calc_xsa(None, init=1)
    df['retxsa'] = df.apply(calc_xsa, axis = 1)
    
    return df['retxsa']    

class PumpDetector(IStrategy):




  
    INTERFACE_VERSION = 2


    stoploss = -0.99

  
    trailing_stop = False
 

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
        
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "XSA": {
                'j': {'color': 'blue'},
                'k': {'color': 'orange'},
            } 
        }
    }
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        

        n1 = 18
        m1 = 4
        m2 = 4

        dataframe['var1']  = dataframe['low'].shift(1)
        dataframe['var1_abs']  = (dataframe['low'] - dataframe['var1']).abs()
        dataframe['var1_max']  = np.where((dataframe['low'] - dataframe['var1']) > 0, (dataframe['low'] - dataframe['var1']), 0)
        dataframe['var2_test'] = xsa(dataframe, source = 'var1_abs', len = 3, wei = 1) 
        dataframe['var2'] = (xsa(dataframe, source = 'var1_abs', len = 3, wei = 1) / xsa(dataframe, source = 'var1_max', len = 3, wei = 1)) * 100
        dataframe['var2_10'] = dataframe['var2'] * 10
        dataframe['var3']  = ta.EMA( dataframe['var2_10'], timeperiod = 3)
        dataframe['var4']  = dataframe['low'].rolling(38).min()
        dataframe['var5']  = dataframe['var3'].rolling(38).max()
        dataframe['var6']  = 1
        dataframe['var7_data']  = np.where(dataframe['low'] <=  dataframe['var4'], (dataframe['var3'] + dataframe['var5'] * 2)/2 , 0 )
        dataframe['var7']  = ta.EMA( dataframe['var7_data'], timeperiod = 3) / 618 * dataframe['var3']

        dataframe['var8']  = ((dataframe['close']-dataframe['low'].rolling(21).min() )/( dataframe['high'].rolling(21).max()  - dataframe['low'].rolling(21).min() ))*100
        dataframe['var9'] = xsa(dataframe, source = 'var8', len = 13, wei = 8)
        
        dataframe['rsv'] = (dataframe['close'] - dataframe['low'].rolling(n1).min() )  /( dataframe['high'].rolling(n1).max() - dataframe['low'].rolling(n1).min()  )*100
        dataframe['k'] = xsa(dataframe, source = 'rsv', len = m1, wei = 1)
        dataframe['d'] = xsa(dataframe, source = 'k', len = m2, wei = 1)
        dataframe['j'] = 3 * dataframe['k'] - 2 * dataframe['d']

        
        # dataframe.to_csv('test.csv')

       

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
     
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['j'],  0)) &   
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
   
        dataframe.loc[
            (
                
                ((qtpylib.crossed_above(dataframe['j'],  90))  |
                (qtpylib.crossed_below(dataframe['j'], dataframe['k']) & dataframe['j'] > 50)  )  & 
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    