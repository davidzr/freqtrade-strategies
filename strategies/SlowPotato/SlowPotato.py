# SlowPotato is a slow order strategy based on averages (5 days) executed on 5m interval 
# the premise is to buy once the average low (5 days) is reached or lower and wait for sell trigger once the average high (5 days) is reached or higher 

# If you want to help with this small endevor please reach me via discord jadex#0557

# If you want to show your support donations are always accepted 
# BTC = 13PustEinvinjud3wCARGHqz34j3GAifjC
# ETH (ERC20) = 0x1b2aaceff8e4475f28280186553c07286e7e3e53



## Suggestions and improvements always welcome.
### Version 0.1
###  to do list / things to be implemented:
### Hyperopt not profitable currently
### sorting of pairs by profit spread
### find a faster/better way to average the High/low for 1 day data for 5 days
### if a day passes after buying with no sell rerun average high based on 5 day average high

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade
from technical.util import resample_to_interval, resampled_merge

import logging
logger = logging.getLogger(__name__)

class SlowPotato(IStrategy):
    """
    This strategy uses the averages for the last 5 days high/low and sets up buy and sell orders acordingly
    Currently developing and testing this strategy
    """

    minimal_roi = { #if you overide this it will sell once it reaches a certain ROI threshold rather than the sell logic
        "0":  99
    }
    
    stoploss = -0.99
    
    # Optimal timeframe for the strategy 
    timeframe = '5m'
    
    # trailing stoploss 
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    
    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True
    
    # Optional order type mapping
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

        
    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
        }
    # run "populate_indicators" only for new candle
    process_only_new_candles = False
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        If close candle breaks lower or equal to average low for last 5 days buy it
        """
        
        dataframe.loc[
            (
                #(dataframe['high'].rolling(1440).mean() / dataframe['low'].rolling(1440).mean() >= 1.05) & ## average spread is #% of profit
                (dataframe['low'] <= dataframe['low'].rolling(1440).mean()) & ## current dataframe is below average low
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        If open candle breaks higher or equal to average high for last 5 days sell it
        """

        dataframe.loc[
            (
                (dataframe['high'] >= dataframe['high'].rolling(1440).mean()) & ## current dataframe is above average high
                (dataframe['volume'] > 0) # volume above zero
            )
        ,'sell'] = 1
        return dataframe