import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series

class Cluc7werk(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00732,
        'bbdelta-tail': 0.94138,
        'close-bblower': 0.0199,
        'closedelta-close': 0.01825,
        'fisher': -0.22987,
        'volume': 16
    }
	
    # Sell hyperspace params:
    sell_params = {
		'sell-bbmiddle-close': 0.99184, 
		'sell-fisher': 0.26832
    }
	
    # ROI table:
    minimal_roi = {
        "0": 0.15373,
        "14": 0.1105,
        "57": 0.08376,
        "147": 0.03427,
        "201": 0.01352,
        "366": 0.00667,
        "469": 0
    }
	
    # Stoploss:
    stoploss = -0.02

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01007
    trailing_stop_positive_offset = 0.01258
    trailing_only_offset_is_reached = False

    """
    END HYPEROPT
    """
    
    timeframe = '1m'

    startup_candle_count: int = 72

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Set Up Bollinger Bands
        upper_bb1, mid_bb1, lower_bb1 = ta.BBANDS(dataframe['close'], timeperiod=40)
        upper_bb2, mid_bb2, lower_bb2 = ta.BBANDS(qtpylib.typical_price(dataframe), timeperiod=20)

        # only putting some bands into dataframe as the others are not used elsewhere in the strategy
        dataframe['lower-bb1'] = lower_bb1
        dataframe['lower-bb2'] = lower_bb2
        dataframe['mid-bb2'] = mid_bb2
       
        dataframe['bb1-delta'] = (mid_bb1 - dataframe['lower-bb1']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        dataframe['ema_fast'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=24).mean()

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=9)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher-rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            (
                dataframe['fisher-rsi'].lt(params['fisher'])
            ) &
            ((      
                    dataframe['bb1-delta'].gt(dataframe['close'] * params['bbdelta-close']) &
                    dataframe['closedelta'].gt(dataframe['close'] * params['closedelta-close']) &
                    dataframe['tail'].lt(dataframe['bb1-delta'] * params['bbdelta-tail']) &
                    dataframe['close'].lt(dataframe['lower-bb1'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift())
            ) |
            (       
                    (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < params['close-bblower'] * dataframe['lower-bb2']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            ((dataframe['close'] * params['sell-bbmiddle-close']) > dataframe['mid-bb2']) &
            dataframe['ema_fast'].gt(dataframe['close']) &
            dataframe['fisher-rsi'].gt(params['sell-fisher']) &
            dataframe['volume'].gt(0)
            ,
            'sell'
        ] = 1

        return dataframe