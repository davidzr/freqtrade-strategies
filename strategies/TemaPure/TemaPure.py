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


class TemaPure(IStrategy):
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
        "0": 0.40505,
        "265": 0.24708,
        "743": 0.05892,
        "1010": 0
    }


    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.09754
    
    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.2126
    trailing_stop_positive_offset = 0.2225
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

        dataframe['CMO'] = ta.CMO(dataframe, timeperiod = 50)
        dataframe['TEMA'] = ta.TEMA(dataframe, timeperiod = 25)
        # Bollinger bands
        
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=25, stds=2.0)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        
        bollingerTA = ta.BBANDS(dataframe, timeperiod=25, nbdevup=3.5, nbdevdn=1.0, matype=0)
        
        dataframe['bb_lowerbandTA'] = bollingerTA['lowerband']
        dataframe['bb_middlebandTA'] = bollingerTA['middleband']
        dataframe['bb_upperbandTA'] = bollingerTA['upperband']

        return dataframe
 
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
                        
        dataframe.loc[
            (
                  # (qtpylib.crossed_below(dataframe["TEMA"], dataframe["bb_lowerbandTA"]))
                  
                         (((dataframe['TEMA']<=dataframe['bb_lowerbandTA']))
            & 
              (qtpylib.crossed_above(dataframe['CMO'],0))) 
                  
                
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
                
                # (qtpylib.crossed_below(dataframe['close'],dataframe['bb_upperbandTA']))
                # & 
                #   (dataframe["CMO"]>=35)
                
            (qtpylib.crossed_above(dataframe["TEMA"], dataframe["bb_upperbandTA"]))
                  
            #   (((dataframe['close'].shift(1)<dataframe['bb_lowerband']))
            # & 
            #   (qtpylib.crossed_above(dataframe['CMO'],0))) 
              
                # | ((qtpylib.crossed_above(dataframe['close'],dataframe['bb_upperbandTA'])))
                
                
            ),
            'sell'] = 1        
        
        return dataframe