import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow
from typing import Dict, List, NamedTuple, Optional, Tuple
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI
from statistics import mean
from cachetools import TTLCache



class SuperHV27(IStrategy):

    timeframe = '5m'

    # Buy hyperspace params:
    buy_params = {
        'adx1': 49,
        'adx2': 36,
        'adx3': 32,
        'adx4': 24,
        'emarsi1': 43,
        'emarsi2': 27,
        'emarsi3': 26,
        'emarsi4': 50
    }

    # Sell hyperspace params:
    sell_params = {
     'adx2': 36, 'emarsi1': 43, 'emarsi2': 27, 'emarsi3': 26
    }

    # ROI table:
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "40": 0.025,
        "60": 0.015,
        "720": 0.01,
        "1440": 0
    }

    # Stoploss:
    stoploss = -0.40

    # connect will participate in hyperopted tables, other methods will not
    dynamic_roi = {
        'enabled': True,
        'type': 'connect',      # linear, exponential, or connect
        'decay-rate': 0.015,    # bigger is faster, recommended to graph f(t) = start-pct * e(-rate*t)      
        'decay-time': 1440,     # amount of time to reach zero, only relevant for linear decay
        'start': 0.10,          # starting percentage
        'end': 0,               # ending percentage 
    }

    # Probably don't change these
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Custom Dicts for storing trade data and other custom things this strategy does
    custom_trade_info = {}
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Populate/update the trade data if there is any, set trades to false if not live/dry
        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])

        dataframe['rsi'] = np.nan_to_num(ta.RSI(dataframe, timeperiod=5))
        dataframe['emarsi'] = np.nan_to_num(ta.EMA(dataframe['rsi'], timeperiod=5))

        dataframe['adx'] = np.nan_to_num(ta.ADX(dataframe))

        dataframe['minusdi'] = np.nan_to_num(ta.MINUS_DI(dataframe))
        dataframe['minusdiema'] = np.nan_to_num(ta.EMA(dataframe['minusdi'], timeperiod=25))
        dataframe['plusdi'] = np.nan_to_num(ta.PLUS_DI(dataframe))
        dataframe['plusdiema'] = np.nan_to_num(ta.EMA(dataframe['plusdi'], timeperiod=5))

        dataframe['lowsma'] = np.nan_to_num(ta.EMA(dataframe, timeperiod=60))
        dataframe['highsma'] = np.nan_to_num(ta.EMA(dataframe, timeperiod=120))
        dataframe['fastsma'] = np.nan_to_num(ta.SMA(dataframe, timeperiod=120))
        dataframe['slowsma'] = np.nan_to_num(ta.SMA(dataframe, timeperiod=240))

        dataframe['bigup'] = dataframe['fastsma'].gt(dataframe['slowsma']) & ((dataframe['fastsma'] - dataframe['slowsma']) > dataframe['close'] / 300)
        dataframe['bigdown'] = ~dataframe['bigup']
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']

        dataframe['preparechangetrend'] = dataframe['trend'].gt(dataframe['trend'].shift())
        dataframe['preparechangetrendconfirm'] = dataframe['preparechangetrend'] & dataframe['trend'].shift().gt(dataframe['trend'].shift(2))
        dataframe['continueup'] = dataframe['slowsma'].gt(dataframe['slowsma'].shift()) & dataframe['slowsma'].shift().gt(dataframe['slowsma'].shift(2))

        dataframe['delta'] = dataframe['fastsma'] - dataframe['fastsma'].shift()
        dataframe['slowingdown'] = dataframe['delta'].lt(dataframe['delta'].shift())

        dataframe['rmi-slow'] = RMI(dataframe, length=21, mom=5)
        dataframe['rmi-fast'] = RMI(dataframe, length=8, mom=4)

        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi-slow'] <= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        if trade_data['active_trade']:
            profit_factor = (1 - (dataframe['rmi-slow'].iloc[-1] / 400))
            rmi_grow = self.linear_growth(30, 70, 180, 720, trade_data['open_minutes'])

            conditions.append(dataframe['rmi-up-trend'] == 1)
            conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * profit_factor))
            conditions.append(dataframe['rmi-slow'] >= rmi_grow)
        
        else:
            # Standard BinHV27 Buy Conditions
            conditions.append(
                dataframe['slowsma'].gt(0) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                (
                (
                    ~dataframe['preparechangetrend'] &
                    ~dataframe['continueup'] &
                    dataframe['adx'].gt(params['adx1']) &
                    dataframe['bigdown'] &
                    dataframe['emarsi'].le(params['emarsi1'])
                ) |
                (
                    ~dataframe['preparechangetrend'] &
                    dataframe['continueup'] &
                    dataframe['adx'].gt(params['adx2']) &
                    dataframe['bigdown'] &
                    dataframe['emarsi'].le(params['emarsi2'])
                ) |
                (
                    ~dataframe['continueup'] &
                    dataframe['adx'].gt(params['adx3']) &
                    dataframe['bigup'] &
                    dataframe['emarsi'].le(params['emarsi3'])
                ) |
                (
                    dataframe['continueup'] &
                    dataframe['adx'].gt(params['adx4']) &
                    dataframe['bigup'] &
                    dataframe['emarsi'].le(params['emarsi4'])
                )
                )
            )

        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # Standard BinHV27 Sell Conditions
        conditions.append(
            (
            (
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                (dataframe['close'].gt(dataframe['lowsma']) | dataframe['close'].gt(dataframe['highsma'])) &
                dataframe['highsma'].gt(0) &
                dataframe['bigdown']
            ) |
            (
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                (dataframe['emarsi'].ge(params['emarsi1']) | dataframe['close'].gt(dataframe['slowsma'])) &
                dataframe['bigdown']
            ) |
            (
                ~dataframe['preparechangetrendconfirm'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                dataframe['adx'].gt(params['adx2']) &
                dataframe['emarsi'].ge(params['emarsi2']) &
                dataframe['bigup']
            ) |
            (
                dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['slowingdown'] &
                dataframe['emarsi'].ge(params['emarsi3']) &
                dataframe['slowsma'].gt(0)
            ) |
            (
                dataframe['preparechangetrendconfirm'] &
                dataframe['minusdi'].lt(dataframe['plusdi']) &
                dataframe['close'].gt(dataframe['lowsma']) &
                dataframe['slowsma'].gt(0)
            )
            )
        )

        if trade_data['active_trade']:  
            loss_cutoff = self.linear_growth(-0.03, 0, 0, 300, trade_data['open_minutes'])

            conditions.append(trade_data['current_profit'] < loss_cutoff)

            if trade_data['other_trades']:
                if trade_data['free_slots'] > 0:
                    hold_pct = (trade_data['free_slots'] / 100) * -1
                    conditions.append(trade_data['avg_other_profit'] >= hold_pct)
                else:
                    conditions.append(trade_data['biggest_loser'] == True)
                           
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe

    """
    Override for default Freqtrade ROI table functionality
    """
    def min_roi_reached_entry(self, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:
        dynamic_roi = self.dynamic_roi

        # if the dynamic_roi dict is defined and enabled, do it, otherwise fallback to default functionality
        if dynamic_roi and dynamic_roi['enabled']:
            # linear decay: f(t) = start - (rate * t)
            if dynamic_roi['type'] == 'linear':
                rate = (dynamic_roi['start'] - dynamic_roi['end']) / dynamic_roi['decay-time']
                min_roi = max(dynamic_roi['end'], dynamic_roi['start'] - (rate * trade_dur))
            # exponential decay: f(t) = start * e^(-rate*t)
            elif dynamic_roi['type'] == 'exponential':
                min_roi = max(dynamic_roi['end'], dynamic_roi['start'] * np.exp(-dynamic_roi['decay-rate']*trade_dur))
            elif dynamic_roi['type'] == 'connect':
                # connect the points in the defined table with lines
                past_roi = list(filter(lambda x: x <= trade_dur, self.minimal_roi.keys()))
                next_roi = list(filter(lambda x: x >  trade_dur, self.minimal_roi.keys()))
                if not past_roi:
                    return None, None
                current_entry = max(past_roi)
                # next entry
                if not next_roi:
                    return current_entry, self.minimal_roi[current_entry]
                # use the slope-intercept formula between the two points in the roi table we are between
                else:
                    next_entry = min(next_roi)
                    # y = mx + b
                    x1 = current_entry
                    x2 = next_entry
                    y1 = self.minimal_roi[current_entry]
                    y2 = self.minimal_roi[next_entry]
                    m = (y1-y2)/(x1-x2)
                    b = (x1*y2 - x2*y1)/(x1-x2)
                    min_roi = (m * trade_dur) + b
            else:
                min_roi = 0

            return trade_dur, min_roi
        # use the standard ROI table
        else:
            # Get highest entry in ROI dict where key <= trade-duration
            roi_list = list(filter(lambda x: x <= trade_dur, self.minimal_roi.keys()))
            if not roi_list:
                return None, None
            roi_entry = max(roi_list)
            return roi_entry, self.minimal_roi[roi_entry]

    """
    Super Legit Custom Methods
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
        return min(end, start + (rate * trade_time))


    """
    Price protection on trade entry and timeouts, built-in Freqtrade functionality
    https://www.freqtrade.io/en/latest/strategy-advanced/
    """
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
Sub-strategy overrides
Anything not explicity defined here will follow the settings in the base strategy
"""
# Sub-strategy with parameters specific to BTC stake
class SuperHV27_BTC(SuperHV27):

    timeframe = '5m'



    use_sell_signal = False

# Sub-strategy with parameters specific to ETH stake
class SuperHV27_ETH(SuperHV27):

    timeframe = '5m'



    use_sell_signal = False