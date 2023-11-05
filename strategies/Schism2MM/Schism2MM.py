import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI
from statistics import mean
from cachetools import TTLCache
from scipy.signal import argrelextrema

class Schism2MM(IStrategy):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'inf-pct-adr': 0.95,
        'inf-rsi': 65,
        'mp': 53,
        'rmi-fast': 41,
        'rmi-slow': 33
    }

    sell_params = {}

    minimal_roi = {
        "0": 0.025,
        "10": 0.015,
        "20": 0.01,
        "30": 0.005,
        "120": 0
    }

    stoploss = -0.99

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 72

    custom_trade_info = {}
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])
    
        dataframe['rmi-slow'] = RMI(dataframe, length=21, mom=5)
        dataframe['rmi-fast'] = RMI(dataframe, length=8, mom=4)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=6)
        dataframe['mp']  = ta.RSI(dataframe['roc'], timeperiod=6)
    
        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi-slow'] <= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['rsi'] = ta.RSI(informative, timeperiod=14)
        informative['1d_high'] = informative['close'].rolling(24).max()
        informative['3d_low'] = informative['close'].rolling(72).min()
        informative['adr'] = informative['1d_high'] - informative['3d_low']
        
        min_peaks = argrelextrema(dataframe['close'].values, np.less, order=100)

        for mp in min_peaks[0]:
             dataframe.at[mp, 'buy_signal'] = True

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        if trade_data['active_trade']:
            rmi_grow = self.linear_growth(30, 70, 180, 720, trade_data['open_minutes'])
            profit_factor = (1 - (dataframe['rmi-slow'].iloc[-1] / 300))
            conditions.append(dataframe['rmi-up-trend'] == 1)
            conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * profit_factor))
            conditions.append(dataframe['rmi-slow'] >= rmi_grow)
        else:
            conditions.append(
                (dataframe[f"rsi_{self.inf_timeframe}"] >= params['inf-rsi']) &
                (dataframe['close'] <= dataframe[f"3d_low_{self.inf_timeframe}"] + (params['inf-pct-adr'] * dataframe[f"adr_{self.inf_timeframe}"])) &
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['rmi-slow'] >= params['rmi-slow']) &
                (dataframe['rmi-fast'] <= params['rmi-fast']) &
                (dataframe['mp'] <= params['mp'])
            )

        conditions.append(dataframe['volume'].gt(0))
        conditions.append(dataframe['buy_signal'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []
        
        if trade_data['active_trade']:
            loss_cutoff = self.linear_growth(-0.03, 0, 0, 300, trade_data['open_minutes'])

            conditions.append(
                (trade_data['current_profit'] < loss_cutoff) & 
                (trade_data['current_profit'] > self.stoploss) &  
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['volume'].gt(0))
            )
            if trade_data['peak_profit'] > 0:
                conditions.append(qtpylib.crossed_below(dataframe['rmi-slow'], 50))
            else:
                conditions.append(qtpylib.crossed_below(dataframe['rmi-slow'], 10))

            if trade_data['other_trades']:
                if trade_data['free_slots'] > 0:
                    hold_pct = (trade_data['free_slots'] / 100) * -1
                    conditions.append(trade_data['avg_other_profit'] >= hold_pct)
                else:
                    conditions.append(trade_data['biggest_loser'] == True)

        else:
            conditions.append(dataframe['volume'].lt(0))
                           
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        
        return dataframe

    def populate_trades(self, pair: str) -> dict:
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        trade_data = {}
        trade_data['active_trade'] = trade_data['other_trades'] = trade_data['biggest_loser'] = False

        if self.config['runmode'].value in ('live', 'dry_run'):
            
            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).all()

            if active_trade:
                current_rate = self.get_current_price(pair, True)
                active_trade[0].adjust_min_max_rates(current_rate)

                present = arrow.utcnow()
                trade_start  = arrow.get(active_trade[0].open_date)
                open_minutes = (present - trade_start).total_seconds() // 60 

                trade_data['active_trade']   = True
                trade_data['current_profit'] = active_trade[0].calc_profit_ratio(current_rate)
                trade_data['peak_profit']    = max(0, active_trade[0].calc_profit_ratio(active_trade[0].max_rate))
                trade_data['open_minutes']   : int = open_minutes
                trade_data['open_candles']   : int = (open_minutes // active_trade[0].timeframe)
            else: 
                trade_data['current_profit'] = trade_data['peak_profit']  = 0.0
                trade_data['open_minutes']   = trade_data['open_candles'] = 0

            other_trades = Trade.get_trades([Trade.pair != pair, Trade.is_open.is_(True),]).all()

            if other_trades:
                trade_data['other_trades'] = True
                other_profit = tuple(trade.calc_profit_ratio(self.get_current_price(trade.pair, False)) for trade in other_trades)
                trade_data['avg_other_profit'] = mean(other_profit) 
                if trade_data['current_profit'] < min(other_profit):
                    trade_data['biggest_loser'] = True
            else:
                trade_data['avg_other_profit'] = 0

            open_trades = len(Trade.get_open_trades())
            trade_data['free_slots'] = max(0, self.config['max_open_trades'] - open_trades)
        return trade_data

    def get_current_price(self, pair: str, refresh: bool) -> float:
        if not refresh:
            rate = self.custom_current_price_cache.get(pair)
            if rate:
                return rate

        ask_strategy = self.config.get('ask_strategy', {})
        if ask_strategy.get('use_order_book', False):
            ob = self.dp.orderbook(pair, 1)
            rate = ob[f"{ask_strategy['price_side']}s"][0][0]
        else:
            ticker = self.dp.ticker(pair)
            rate = ticker['last']

        self.custom_current_price_cache[pair] = rate
        return rate

    def linear_growth(self, start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
        time = max(0, trade_time - start_time)
        rate = (end - start) / (end_time - start_time)
        return min(end, start + (rate * trade_time))

    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        if current_price > order['price'] * 1.01:
            return True
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ask_strategy = self.config.get('ask_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{ask_strategy['price_side']}s"][0][0]
        if current_price < order['price'] * 0.99:
            return True
        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        if current_price > rate * 1.01:
            return False
        return True