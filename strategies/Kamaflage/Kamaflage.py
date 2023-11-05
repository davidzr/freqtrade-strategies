# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI, VIDYA


class Kamaflage(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    buy_params = {
        'macd': 0,
        'macdhist': 0,
        'rmi': 50
    }

    sell_params = {

    }

    minimal_roi = {
        "0": 0.15,
        "10": 0.10,
        "20": 0.05,
        "30": 0.025,
        "60": 0.01
    }

    # Stoploss:
    stoploss = -1

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01125
    trailing_stop_positive_offset = 0.04673
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    # sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    process_only_new_candles = False

    startup_candle_count: int = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:        

        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['rmi'] = RMI(dataframe)
        dataframe['kama-3'] = ta.KAMA(dataframe, timeperiod=3)
        dataframe['kama-21'] = ta.KAMA(dataframe, timeperiod=21)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['volume_ma'] = dataframe['volume'].rolling(window=24).mean()

        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        conditions = []

        active_trade = False
        if self.config['runmode'].value in ('live', 'dry_run'):
            active_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True),]).all()

        if not active_trade:     
            conditions.append(dataframe['kama-3'] > dataframe['kama-21'])
            conditions.append(dataframe['macd'] > dataframe['macdsignal'])
            conditions.append(dataframe['macd'] > params['macd'])
            conditions.append(dataframe['macdhist'] > params['macdhist'])
            conditions.append(dataframe['rmi'] > dataframe['rmi'].shift())
            conditions.append(dataframe['rmi'] > params['rmi'])
            conditions.append(dataframe['volume'] < (dataframe['volume_ma'] * 20))
        else:
            conditions.append(dataframe['close'] > dataframe['sar'])
            conditions.append(dataframe['rmi'] >= 75)

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params
        conditions = []

        active_trade = False
        if self.config['runmode'].value in ('live', 'dry_run'):
            active_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True),]).all()
        
        if active_trade:
            ob = self.dp.orderbook(metadata['pair'], 1)
            current_price = ob['asks'][0][0]
            current_profit = active_trade[0].calc_profit_ratio(rate=current_price)

            conditions.append(
                (dataframe['buy'] == 0) &
                (dataframe['rmi'] < 30) &
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