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

"""
    https://fr.tradingview.com/script/dV5HEGpP-Ultimate-Momentum-Indicator-CC/
    translated for freqtrade: viksal1982  viktors.s@gmail.com
"""
 


class UltimateMomentumIndicator(IStrategy):
  
    


    INTERFACE_VERSION = 2


 

    length1_buy = IntParameter(2, 20, default= 13, space='buy')
    length2_buy = IntParameter(10, 40, default= 19, space='buy')
    length3_buy = IntParameter(10, 50, default= 21, space='buy')
    length4_buy = IntParameter(20, 80, default= 39, space='buy')
    length5_buy = IntParameter(30, 100, default= 50, space='buy')
    length6_buy = IntParameter(150, 300, default= 200, space='buy')
   
    stoploss = -0.99
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".


    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    custom_info = {}
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

            "utmi": {
                'utmi': {'color': 'red'},
            }
        }
    }


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        source = 'close'
        length6 = int(self.length6_buy.value)
        length1 = int(self.length1_buy.value)
        length2 = int(self.length2_buy.value)
        length3 = int(self.length3_buy.value)
        length4 = int(self.length4_buy.value)
        length5 = int(self.length5_buy.value)
        dataframe['basis'] = ta.SMA(dataframe[source], timeperiod = length6)
        dataframe['dev'] = dataframe[source].rolling(length6).std()
        dataframe['upperBand'] = dataframe['basis'] + dataframe['dev']
        dataframe['lowerBand'] = dataframe['basis'] - dataframe['dev']
        dataframe['bPct'] = np.where( (dataframe['upperBand'] - dataframe['lowerBand'] ) != 0, (dataframe[source] - dataframe['lowerBand'])/(dataframe['upperBand'] - dataframe['lowerBand']),0 )

        dataframe['advSum'] = pd.Series(np.where(dataframe[source].diff() > 0, 1, 0)).rolling(length2).sum()
        dataframe['decSum'] = pd.Series(np.where(dataframe[source].diff() > 0, 0, 1)).rolling(length2).sum()
        dataframe['ratio'] = np.where(dataframe['decSum'] != 0, dataframe['advSum']/dataframe['decSum'], 0)
        dataframe['rana'] = np.where( (dataframe['advSum'] + dataframe['decSum']) != 0, (dataframe['advSum'] - dataframe['decSum'])/(dataframe['advSum'] + dataframe['decSum']), 0)
        dataframe['mo'] = ta.EMA(dataframe['rana'], timeperiod = length2) - ta.EMA(dataframe['rana'], timeperiod = length4) 
        dataframe['utm'] = (200 * dataframe['bPct']) + (100 * dataframe['ratio']) + (2 * dataframe['mo']) + (1.5 * ta.MFI(dataframe, timeperiod = length5) ) + (3 * ta.MFI(dataframe, timeperiod = length3) ) + (3 * ta.MFI(dataframe, timeperiod = length1) ) 
        dataframe['utmiRsi']  = ta.RSI(dataframe['utm'], timeperiod = length1)
        dataframe['utmi'] = ta.EMA(dataframe['utmiRsi'], timeperiod = length1)

        

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                 (qtpylib.crossed_above(dataframe['utmi'], 50)) &  
                 (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
             
                (qtpylib.crossed_below(dataframe['utmi'], 70)) &  
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    