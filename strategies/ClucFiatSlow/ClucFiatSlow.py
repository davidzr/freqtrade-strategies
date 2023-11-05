import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from datetime import datetime
from freqtrade.persistence import Trade
from pandas import DataFrame, Series

class ClucFiatSlow(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00642,
        'bbdelta-tail': 0.75559,
        'close-bblower': 0.01415,
        'closedelta-close': 0.00883,
        'fisher': -0.97101,
        'volume': 18
    }
	
    # Sell hyperspace params:
    sell_params = {
		'sell-bbmiddle-close': 0.95153, 
		'sell-fisher': 0.60924
    }

    # ROI table:
    minimal_roi = {
        "0": 0.04354,
        "5": 0.03734,
        "8": 0.02569,
        "10": 0.019,
        "76": 0.01283,
        "235": 0.007,
        "415": 0
    }
	
    # Stoploss:
    stoploss = -0.34299
	
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01057
    trailing_stop_positive_offset = 0.03668
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """
    
    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 48

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

    """
    https://www.freqtrade.io/en/latest/strategy-advanced/

    Custom Order Timeouts
    """
    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['bids'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > order['price'] * 1.01:
            return True
        return False


    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel sell order if price is more than 1% below the order.
        if current_price < order['price'] * 0.99:
            return True
        return False

