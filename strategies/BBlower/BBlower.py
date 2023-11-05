#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:50:49 2020

@author: alex
"""

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class BBlower(IStrategy):
    """
    Sample strategy implementing Informative Pairs - compares stake_currency with USDT.
    Not performing very well - but should serve as an example how to use a referential pair against USDT.
    author@: xmatthias
    github@: https://github.com/freqtrade/freqtrade-strategies
    How to use it?
    > python3 freqtrade -s InformativeSample
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    # ROI table:
    minimal_roi = {
        "0": 0.32477,
        "220": 0.13561,
        "962": 0.10732,
        "2115": 0
    }


    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.13912
    
    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.29846
    trailing_stop_positive_offset = 0.30425
    trailing_only_offset_is_reached = True

    # run "populate_indicators" only for new candle
    ta_on_candle = False

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False


    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return [(f"{self.config['stake_currency']}/USDT", self.timeframe)]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        dataframe['CMO'] = ta.CMO(dataframe, timeperiod = 25)
        dataframe['RSI'] = ta.RSI(dataframe, timeperiod = 25)
        dataframe['TEMA'] = ta.TEMA(dataframe, timeperiod = 50)
      
        # Bollinger bands
        bollingerTA1 = ta.BBANDS(dataframe, timeperiod=25, nbdevup=1.0, nbdevdn=1.0, matype=0)
        
        dataframe['bb_lowerbandTA1'] = bollingerTA1['lowerband']
        dataframe['bb_middlebandTA1'] = bollingerTA1['middleband']
        dataframe['bb_upperbandTA1'] = bollingerTA1['upperband']
        
        bollingerTA2 = ta.BBANDS(dataframe, timeperiod=25, nbdevup=2.0, nbdevdn=2.0, matype=0)
        
        dataframe['bb_lowerbandTA2'] = bollingerTA2['lowerband']
        dataframe['bb_middlebandTA2'] = bollingerTA2['middleband']
        dataframe['bb_upperbandTA2'] = bollingerTA2['upperband']
        
        bollingerTA3 = ta.BBANDS(dataframe, timeperiod=25, nbdevup=3.0, nbdevdn=3.0, matype=0)
        
        dataframe['bb_lowerbandTA3'] = bollingerTA3['lowerband']
        dataframe['bb_middlebandTA3'] = bollingerTA3['middleband']
        dataframe['bb_upperbandTA3'] = bollingerTA3['upperband']
        
        bollingerTA4 = ta.BBANDS(dataframe, timeperiod=25, nbdevup=4.0, nbdevdn=4.0, matype=0)
        
        dataframe['bb_lowerbandTA4'] = bollingerTA4['lowerband']
        dataframe['bb_middlebandTA4'] = bollingerTA4['middleband']
        dataframe['bb_upperbandTA4'] = bollingerTA4['upperband']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
                        
        dataframe.loc[
            (
                # (dataframe['TEMA'] < dataframe['bb_upperbandTA1'])
                # &
                (dataframe['RSI'] > dataframe['RSI'].shift(1))
                &
                (dataframe['RSI'].shift(1) > dataframe['RSI'].shift(2))
                &
                (dataframe['RSI'].shift(2) > dataframe['RSI'].shift(3))
                &
                (dataframe['RSI'].shift(3) > dataframe['RSI'].shift(4))
                &
                (dataframe['RSI'] < 50)
                &
                (qtpylib.crossed_above(
                        dataframe['TEMA'], dataframe['bb_lowerbandTA1']
                    ))
                  
                
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
                       
        dataframe.loc[
            (
                
            #     (dataframe['CMO'] > 23) 
            # & 
            #   (dataframe['RSI'] > 98) 
            #       &
                # (dataframe['TEMA'] > dataframe['bb_upperbandTA2'])
           
                
                
            ),
            'sell'] = 1        
        
        return dataframe