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
from statistics import mean

"""
TODO: 
    - Better buy signal.
    - Potentially leverage an external data source?
"""

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

class Fakebuy(IStrategy):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'bbdelta-close': 0.01697,
        'bbdelta-tail': 0.85522,
        'close-bblower': 0.01167,
        'closedelta-close': 0.00513,
        'rocr-1h': 0.54614,
        'volume': 32
    }

    # ROI table:
    minimal_roi = {
        "0": 0.15,
        "5": 0.025,
        "10": 0.015,
        "30": 0.005
    }

    # Stoploss:
    stoploss = -0.085

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 168

    custom_trade_info = {}

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Misc. calculations regarding existing open positions (reset on every loop iteration)
        self.custom_trade_info[metadata['pair']] = trade_data = {}
        trade_data['active_trade'] = trade_data['other_trades'] = False

        if self.config['runmode'].value in ('live', 'dry_run'):
            
            active_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True),]).all()
            other_trades = Trade.get_trades([Trade.pair != metadata['pair'], Trade.is_open.is_(True),]).all()

            if active_trade:
                current_rate = self.get_current_price(metadata['pair'])
                active_trade[0].adjust_min_max_rates(current_rate)
                trade_data['active_trade']   = True
                trade_data['current_profit'] = active_trade[0].calc_profit_ratio(current_rate)
                trade_data['peak_profit']    = active_trade[0].calc_profit_ratio(active_trade[0].max_rate)

            if other_trades:
                trade_data['other_trades'] = True
                total_other_profit = tuple(trade.calc_profit_ratio(self.get_current_price(trade.pair)) for trade in other_trades)
                trade_data['avg_other_profit'] = mean(total_other_profit) 

        self.custom_trade_info[metadata['pair']] = trade_data

        # Set up Bollinger Bands
        mid, lower = bollinger_bands(dataframe['close'], window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb-lowerband'] = bollinger['lower']
        dataframe['bb-middleband'] = bollinger['mid']
    
        # Set up other indicators
        dataframe['volume-mean-slow'] = dataframe['volume'].rolling(window=24).mean()
        dataframe['rmi-slow'] = RMI(dataframe, length=20, mom=5)
        dataframe['rmi-fast'] = RMI(dataframe, length=9, mom=3)
        dataframe['rocr'] = ta.ROCR(dataframe, timeperiod=28)
        dataframe['ema-slow'] = ta.EMA(dataframe, timeperiod=50)

        # Trend Calculations
        dataframe['max'] = dataframe['high'].rolling(12).max()      
        dataframe['min'] = dataframe['low'].rolling(12).min()       
        dataframe['upper'] = np.where(dataframe['max'] > dataframe['max'].shift(),1,0)      
        dataframe['lower'] = np.where(dataframe['min'] < dataframe['min'].shift(),1,0)      
        dataframe['up_trend'] = np.where(dataframe['upper'].rolling(3, min_periods=1).sum() != 0,1,0)      
        dataframe['dn_trend'] = np.where(dataframe['lower'].rolling(3, min_periods=1).sum() != 0,1,0)

        # Informative Pair Indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['rocr'] = ta.ROCR(informative, timeperiod=168) 

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # Persist a buy signal for existing trades to make use of ignore_roi_if_buy_signal = True
        # when this buy signal is not present a sell can happen according to ROI table
        if trade_data['active_trade']:
            if (trade_data['peak_profit'] > 0):
                conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * 0.8))
            conditions.append(dataframe['rmi-slow'] >= 60)

        # Normal buy triggers that apply to new trades we want to enter
        else:
            dataframe.loc[
                (
                    (dataframe['rocr_1h'] > params['rocr-1h'])
                ) &
                ((      
                        (dataframe['lower'].shift() > 0) &
                        (dataframe['bbdelta'] > (dataframe['close'] * params['bbdelta-close'])) &
                        (dataframe['closedelta'] > (dataframe['close'] * params['closedelta-close'])) &
                        (dataframe['tail'] < (dataframe['bbdelta'] * params['bbdelta-tail'])) &
                        (dataframe['close'] < dataframe['lower'].shift()) &
                        (dataframe['close'] <= dataframe['close'].shift())
                ) |
                (       
                        (dataframe['close'] < dataframe['ema-slow']) &
                        (dataframe['close'] < params['close-bblower'] * dataframe['bb-lowerband']) &
                        (dataframe['volume'] < (dataframe['volume-mean-slow'].shift(1) * params['volume']))
                )),
                'fake_buy'
            ] = 1

            conditions.append(dataframe['fake_buy'].shift(1).eq(1))        
            conditions.append(dataframe['fake_buy'].eq(1))

        # applies to both new buys and persisting buy signal
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []
        
        # if we are in an active trade for this pair
        if trade_data['active_trade']:
            # if we are at a loss, consider what the trend looks and preempt the stoploss
            conditions.append(
                (trade_data['current_profit'] < 0) &
                (trade_data['current_profit'] > self.stoploss) &  
                (dataframe['dn_trend'] == 1) &
                (dataframe['rmi-fast'] < 50) &
                (dataframe['volume'].gt(0))
            )

            # if there are other open trades in addition to this one, consider the average profit 
            # across them all (not including this one), don't sell if entire market is down big and wait for recovery
            if trade_data['other_trades']:
                conditions.append(trade_data['avg_other_profit'] >= -0.005)

        # the bot comes through this loop even when there isn't an open trade to sell
        # so we pass an impossible condiiton here because we don't want a sell signal 
        # clogging up the charts and not having one leads the bot to crash
        else:
            conditions.append(dataframe['volume'].lt(0))
                           
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        
        return dataframe

    """
    Custom methods
    """
    def get_current_price(self, pair: str) -> float:
        """
        # Using ticker seems significantly faster than orderbook.
        side = "asks"
        if (self.config['ask_strategy']['price_side'] == "bid"):
            side = "bids"
        
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[side][0][0]
        """

        ticker = self.dp.ticker(pair)
        current_price = ticker['last']

        return current_price

    """
    Price protection on trade entry and timeouts, built-in Freqtrade functionality
    https://www.freqtrade.io/en/latest/strategy-advanced/
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

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > rate * 1.01:
            return False
        return True


