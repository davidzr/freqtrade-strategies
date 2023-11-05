# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from datetime import timedelta, datetime, timezone
#from freqtrade.strategy.strategy_helper import  merge_informative_pair
from typing import Dict, List
import numpy as np
# --------------------------------
# 11-Aug-20  - seems to be good making a few trades  5 days 33 wins 7 losses AVE 0.41% tot ROI 17.14%

class cryptohassle(IStrategy):
    """

    author@: Sp0ngeB0bUK
    Title:  Crypto Hassle 
    Version: 0.1

    Heikin Ashi Candles - SSL Channel, Momentum cross supported by MACD
    
    """
    
    
   
    minimal_roi = {
        "0": 0.50,
        #"192": -1
        
    }

    # Stoploss:
    stoploss = -0.20
    
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.07
    trailing_only_offset_is_reached = True
    
    


    # Optimal ticker interval for the strategy
    ticker_interval = '1h'
    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    plot_config = {
        'main_plot': {
            # Configuration for main plot indicators.
            # Specifies `ema10` to be red, and `ema50` to be a shade of gray
            'ha_ema9': {'color': 'green'},
            'ha_ema20': {'color': 'red'}
        },
        'subplots': {
            # Additional subplot RSI
            "ADX": {
                'ha_adx': {'color': 'blue'}
            }
        }
    }
    
    
       
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        # Heikinashi EMA
        #dataframe['ha_ema9'] = ta.EMA(heikinashi, timeperiod=9)
        #dataframe['ha_ema20'] = ta.EMA(heikinashi, timeperiod=20)
        # Heikinashi ADX
        #dataframe['ha_adx'] = ta.ADX(heikinashi)
        # HeikinAshi EMA cross to ADX cross rolling delta
        #dataframe['ha_adx_cross'] = qtpylib.crossed_above(dataframe['ha_adx'],25)
        # HeikinAhi EMA9 crossed above EMA20
        #dataframe['ha_ema_cross_above'] = qtpylib.crossed_above(dataframe['ha_ema9'],dataframe['ha_ema20'])
        # HeikinAhi EMA9 crossed below EMA20
        #dataframe['ha_ema_cross_below'] = qtpylib.crossed_below(dataframe['ha_ema9'],dataframe['ha_ema20'])
        # Heikin Ashi Momentum
        #Momentum
        dataframe['ha_mom'] = ta.MOM(heikinashi, timeperiod=14)
        dataframe['ha_mom_cross_above'] = qtpylib.crossed_above(dataframe['ha_mom'],0)
        # Heikin Ashi Candles
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']  
        dataframe['ha_high'] = heikinashi['high'] 
        dataframe['ha_low'] = heikinashi['low']
        # Heikin Ashi MACD
        macd = ta.MACD(heikinashi)
        dataframe['ha_macd'] = macd['macd']
        dataframe['ha_macdsignal'] = macd['macdsignal']
        dataframe['ha_macdhist'] = macd['macdhist']
        dataframe['ha_macd_cross_above'] = qtpylib.crossed_above(dataframe['ha_macd'],dataframe['ha_macdsignal'])
        # Heikin Ashi SSl Channels
        def SSLChannels(dataframe, length = 10, mode='sma'):
            """
            Source: https://www.tradingview.com/script/xzIoaIJC-SSL-channel/
            Author: xmatthias
            Pinescript Author: ErwinBeckers
            SSL Channels.
            Average over highs and lows form a channel - lines "flip" when close crosses either of the 2 lines.
            Trading ideas:
                * Channel cross
                * as confirmation based on up > down for long
                MC - MODIFIED FOR HA CANDLES
            """
            if mode not in ('sma'):
                raise ValueError(f"Mode {mode} not supported yet")
            df = dataframe.copy()
            if mode == 'sma':
                df['smaHigh'] = df['ha_high'].rolling(length).mean()
                df['smaLow'] = df['ha_low'].rolling(length).mean()
            df['hlv'] = np.where(df['ha_close'] > df['smaHigh'], 1, np.where(df['ha_close'] < df['smaLow'], -1, np.NAN))
            df['hlv'] = df['hlv'].ffill()
            df['ha_sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
            df['ha_sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
            return df['ha_sslDown'], df['ha_sslUp']
        ssl = SSLChannels(dataframe, 10)
        dataframe['ha_sslDown'] = ssl[0]
        dataframe['ha_sslUp'] = ssl[1]
        dataframe['ha_ssl_cross_above'] = qtpylib.crossed_above(dataframe['ha_sslUp'],dataframe['ha_sslDown'])
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    
                    # Heikin Ashi SSL Channels
                    (dataframe['ha_ssl_cross_above'].rolling(5).apply(lambda x: x.any(), raw=False) == 1) &
                    # Momentum
                    (dataframe['ha_mom_cross_above'].rolling(5).apply(lambda x: x.any(), raw=False) == 1) &
                    # Heikin Ashi MacD
                    (dataframe['ha_macd_cross_above'].rolling(5).apply(lambda x: x.any(), raw=False) == 1) &
                    # Volume
                    (dataframe['volume'] > 1000)
                    
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                
                qtpylib.crossed_below(dataframe['ha_sslUp'],dataframe['ha_sslDown']) &
                (dataframe['volume'] > 0)
                              
            ),
        'sell'] = 1
        return dataframe