import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from technical.indicators import RMI
from statistics import mean
from cachetools import TTLCache


class Schism5(IStrategy):
    """
    Strategy Configuration Items
    """
    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'bear-buy-rsi': 49, 
        'bull-buy-rsi': 39
    }

    sell_params = {
        'bear-sell-rsi': 86, 
        'bull-sell-rsi': 86
    }

    minimal_roi = {
        "0": 0.14025,
        "34": 0.08031,
        "86": 0.03995,
        "203": 0
    }

    stoploss = -0.30
    use_custom_stoploss = True
    custom_stop_ramp_minutes = 110
    custom_stop_trailing = 0.001

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 72

    custom_trade_info = {}
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
    
    """
    Informative Pair Definitions
    """
    def informative_pairs(self):
        # add existing pairs from whitelist on the inf_timeframe
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        return informative_pairs

    """
    Indicator Definitions
    """ 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])
    
        dataframe['rmi-slow'] = RMI(dataframe, length=21, mom=5)
        dataframe['rmi-fast'] = RMI(dataframe, length=8, mom=4)
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
   
        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi-slow'] <= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['rsi'] = ta.RSI(informative, timeperiod=14)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        dataframe['bull'] = dataframe[f"rsi_{self.inf_timeframe}"].gt(60).astype('int') * 20

        return dataframe

    """
    Buy Trigger Signals
    """
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.get_pair_params(metadata['pair'], 'buy')
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        if trade_data['active_trade']:
            profit_factor = (1 - (dataframe['rmi-slow'].iloc[-1] / 400))
            rmi_grow = self.linear_growth(30, 70, 0, 240, trade_data['open_minutes'])

            conditions.append(dataframe['rmi-up-trend'] == 1)
            conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * profit_factor))
            conditions.append(dataframe['rmi-slow'] >= rmi_grow)

        else:
 
             conditions.append(
                ((dataframe['bull'] > 0) & qtpylib.crossed_below(dataframe['rsi'], params['bull-buy-rsi'])) |
                (~(dataframe['bull'] > 0) & qtpylib.crossed_below(dataframe['rsi'], params['bear-buy-rsi']))
            )
            
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    """
    Sell Trigger Signals
    """
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.get_pair_params(metadata['pair'], 'sell')
        conditions = []

        conditions.append(
            ((dataframe['bull'] > 0) & (dataframe['rsi'] > params['bull-sell-rsi'])) |
            (~(dataframe['bull'] > 0) & (dataframe['rsi'] > params['bear-sell-rsi']))
        )

        conditions.append(dataframe['volume'].gt(0))
            
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:

        since_open = current_time - trade.open_date

        sl_pct = 1 - (max(min(since_open / timedelta(minutes=self.custom_stop_ramp_minutes), 1), 0))**3
        sl_ramp = self.stoploss * sl_pct

        return min(0, sl_ramp) - self.custom_stop_trailing

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

    """
    Custom Methods
    """
    def populate_trades(self, pair: str) -> dict:
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        trade_data = {}
        trade_data['active_trade'] = trade_data['other_trades'] = trade_data['biggest_loser'] = False
        self.custom_trade_info['meta'] = {}

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
        return min(end, start + (rate * time))

    def get_pair_params(self, pair: str, side: str) -> Dict:
        buy_params = self.buy_params
        sell_params = self.sell_params
  
        ### Stake: USD
        if pair in ('ABC/XYZ', 'DEF/XYZ'):
            buy_params = self.buy_params_GROUP1
            sell_params = self.sell_params_GROUP1
        elif pair in ('QRD/WTF'):
            buy_params = self.buy_params_QRD
            sell_params = self.sell_params_QRD

        if side == 'sell':
            return sell_params

        return buy_params


class Schism5_BTC(Schism5):

    timeframe = '1h'
    inf_timeframe = '4h'

    buy_params = {
        'inf-rsi': 64,
        'mp': 55,
        'rmi-fast': 31,
        'rmi-slow': 16,
        'xinf-stake-rmi': 67,
        'xtf-fiat-rsi': 17,
        'xtf-stake-rsi': 57
    }

    minimal_roi = {
        "0": 0.05,
        "240": 0.025,
        "1440": 0.01,
        "4320": 0
    }

    use_sell_signal = False

class Schism5_ETH(Schism5):

    timeframe = '1h'
    inf_timeframe = '4h'

    buy_params = {
        'inf-rsi': 13,
        'inf-stake-rmi': 69,
        'mp': 40,
        'rmi-fast': 42,
        'rmi-slow': 17,
        'tf-fiat-rsi': 15,
        'tf-stake-rsi': 92
    }

    minimal_roi = {
        "0": 0.05,
        "240": 0.025,
        "1440": 0.01,
        "4320": 0
    }

    use_sell_signal = False