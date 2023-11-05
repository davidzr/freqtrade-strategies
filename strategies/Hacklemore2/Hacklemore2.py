import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI

class Hacklemore2(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """
    # ROI table:
    minimal_roi = {
        "0": 0.14509,
        "9": 0.07666,
        "23": 0.0378,
        "36": 0.01987,
        "60": 0.0128,
        "145": 0.00467,
        "285": 0
    }

    # Stoploss:
    stoploss = -0.99

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """
    
    timeframe = '15m'

    use_sell_signal = True
    sell_profit_only = False
    #sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=24).mean()

        dataframe['RMI'] = RMI(dataframe)
        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['max'] = dataframe['high'].rolling(60).max()      

        dataframe['min'] = dataframe['low'].rolling(60).min()       

        dataframe['upper'] = np.where(dataframe['max'] > dataframe['max'].shift(),1,0)      

        dataframe['lower'] = np.where(dataframe['min'] < dataframe['min'].shift(),1,0)      

        dataframe['up_trend'] = np.where(dataframe['upper'].rolling(10, min_periods=1).sum() != 0,1,0)      

        dataframe['dn_trend'] = np.where(dataframe['lower'].rolling(10, min_periods=1).sum() != 0,1,0)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        active_trade = False

        if self.config['runmode'].value in ('live', 'dry_run'):
            active_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True),]).all()

        conditions = []

        if not active_trade:
            conditions.append(
            (   (dataframe['up_trend'] == True) &
                (dataframe['RMI'] > 55) &
                (dataframe['RMI'] >= dataframe['RMI'].rolling(3).mean()) &
                (dataframe['close'] > dataframe['close'].shift()) &
                (dataframe['close'].shift() > dataframe['close'].shift(2)) &
                (dataframe['sar'] < dataframe['close']) &
                (dataframe['sar'].shift() < dataframe['close'].shift()) &
                (dataframe['sar'].shift(2) < dataframe['close'].shift(2)) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 30))
            ))

        else:
            conditions.append(dataframe['RMI'] >= 75) 

        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        active_trade = False

        if self.config['runmode'].value in ('live', 'dry_run'):
            active_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True),]).all()
        
        conditions = []

        if active_trade:
            ob = self.dp.orderbook(metadata['pair'], 1)
            current_price = ob['asks'][0][0]
            # current_profit = Trade.calc_profit_ratio(active_trade[0], rate=current_price)
            current_profit = active_trade[0].calc_profit_ratio(rate=current_price)

            conditions.append(
                (dataframe['buy'] == 0) &
                (dataframe['dn_trend'] == True) &
                (dataframe['RMI'] < 30) &
                (current_profit > -0.03) &
                (dataframe['volume'].gt(0))
            ) 
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        else:
            dataframe['sell'] = 0
      
        return dataframe

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

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > rate * 1.01:
            return False
        return True

    """
    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        _, roi = self.min_roi_reached_entry(0)

        if roi is None:
           if Trade.max_rate >= Trade.rate * 0.8 and Trade.rate > Trade.open_rate: 
                return False
            if Trade.max_rate < Trade.rate * 0.8 and Trade.rate < Trade.open_rate: 
                return False
            if Trade.max_rate < Trade.rate * 0.8 and Trade.rate > Trade.open_rate: 
                return current_profit > roi
        return False
    """