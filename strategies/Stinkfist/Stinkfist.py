import numpy as np
import talib.abstract as ta
import technical.indicators as ti
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from statistics import mean
from cachetools import TTLCache
from collections import namedtuple


"""
Loosely based on:
https://github.com/nicolay-zlobin/jesse-indicators/blob/main/strategies/BadStreak/__init__.py

TODO:
    - Move custom indicators out to helper file or freqtrade/technical?

"""

class Stinkfist(IStrategy):
    """
    Strategy Configuration Items
    """
    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'inf-pct-adr': 0.80,
        'mp': 30
    }

    sell_params = {}

    minimal_roi = {
        "0": 0.05,
        "10": 0.025,
        "20": 0.015,
        "30": 0.01,
        "720": 0.005,
        "1440": 0
    }

    stoploss = -0.40

    # Probably don't change these
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 72

    # Custom Dicts for storing trade data and other custom things this strategy does
    custom_trade_info = {}
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes
    
    """
    Informative Pair Definitions
    """
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        return informative_pairs

    """
    Indicator Definitions
    """ 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Populate/update the trade data if there is any, set trades to false if not live/dry
        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])
    
        # Set up primary indicators
        dataframe['rmi-slow'] = ti.RMI(dataframe, length=21, mom=5)
        dataframe['rmi-fast'] = ti.RMI(dataframe, length=8, mom=4)

        # MA Streak
        dataframe['mac'] = self.mac(dataframe, 20, 50)
        dataframe['streak'] = self.ma_streak(dataframe, period=4)

        streak = abs(int(dataframe['streak'].iloc[-1]))
        streak_back_close = dataframe['close'].shift(streak + 1)

        dataframe['streak-roc'] = 100 * (dataframe['close'] - streak_back_close) / streak_back_close

        # Percent Change Channel
        pcc = self.pcc(dataframe, period=20, mult=2)
        dataframe['pcc-lowerband'] = pcc.lowerband

        # Momentum Pinball
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['mp']  = ta.RSI(dataframe['roc'], timeperiod=3)

        # Trend Calculations    
        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi-slow'] <= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

        # Informative indicators for current pair on inf_timeframe
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['1d_high'] = informative['close'].rolling(24).max()
        informative['3d_low'] = informative['close'].rolling(72).min()
        informative['adr'] = informative['1d_high'] - informative['3d_low']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    """
    Buy Trigger Signals
    """
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
            # Primary buy triggers
            conditions.append(
                # informative timeframe conditions
                (dataframe['close'] <= dataframe[f"3d_low_{self.inf_timeframe}"] + (params['inf-pct-adr'] * dataframe[f"adr_{self.inf_timeframe}"])) &
                # default timeframe conditions
                (dataframe['mp'] < params['mp']) &
                (dataframe['streak-roc'] > dataframe['pcc-lowerband']) &
                (dataframe['mac'] == 1)
            )

        # Anything below here applies to persisting and new buy signal
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
                conditions.append(dataframe['rmi-slow'] < 50)
            else:
                conditions.append(dataframe['rmi-slow'] < 10)

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
    Custom Indicators
    """

    """
    Moving Average Cross
    Port of: https://www.tradingview.com/script/PcWAuplI-Moving-Average-Cross/
    """
    def mac(self, dataframe: DataFrame, fast: int = 20, slow: int = 50) -> Series:

        dataframe = dataframe.copy()

        # Fast MAs
        upper_fast = ta.EMA(dataframe['high'], timeperiod=fast)
        lower_fast = ta.EMA(dataframe['low'], timeperiod=fast)

        # Slow MAs
        upper_slow = ta.EMA(dataframe['high'], timeperiod=slow)
        lower_slow = ta.EMA(dataframe['low'], timeperiod=slow)

        # Crosses
        crosses_lf_us = qtpylib.crossed_above(lower_fast, upper_slow) | qtpylib.crossed_below(lower_fast, upper_slow)
        crosses_uf_ls = qtpylib.crossed_above(upper_fast, lower_slow) | qtpylib.crossed_below(upper_fast, lower_slow)

        dir_1 = np.where(crosses_lf_us, 1, np.nan)
        dir_2 = np.where(crosses_uf_ls, -1, np.nan)

        dir = np.where(dir_1 == 1, dir_1, np.nan)
        dir = np.where(dir_2 == -1, dir_2, dir_1)

        res = Series(dir).fillna(method='ffill').to_numpy()

        return res

    """
    MA Streak
    Port of: https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/
    """
    def ma_streak(self, dataframe: DataFrame, period: int = 4, source_type='close') -> Series:
        
        dataframe = dataframe.copy()

        avgval = self.zlema(dataframe[source_type], period)

        arr = np.diff(avgval)
        pos = np.clip(arr, 0, 1).astype(bool).cumsum()
        neg = np.clip(arr, -1, 0).astype(bool).cumsum()
        streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),
                        -neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))

        res = np.concatenate((np.full((dataframe.shape[0] - streak.shape[0]), np.nan), streak))

        return res

    """
    Percent Change Channel
    PCC is like KC unless it uses percentage changes in price to set channel distance.
    https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/
    """
    def pcc(self, dataframe: DataFrame, period: int = 20, mult: int = 2):

        PercentChangeChannel = namedtuple('PercentChangeChannel', ['upperband', 'middleband', 'lowerband'])

        dataframe = dataframe.copy()

        close = dataframe['close']
        previous_close = close.shift()
        low = dataframe['low']
        high = dataframe['high']

        close_change = (close - previous_close) / previous_close * 100
        high_change = (high - close) / close * 100
        low_change = (low - close) / close * 100

        mid = self.zlema(close_change, period)
        rangema = self.zlema(high_change - low_change, period)

        upper = mid + rangema * mult
        lower = mid - rangema * mult

        return PercentChangeChannel(upper, rangema, lower)

    def zlema(self, series: Series, period):
        ema1 = ta.EMA(series, period)
        ema2 = ta.EMA(ema1, period)
        d = ema1 - ema2
        zlema = ema1 + d
        return zlema

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
class Stinkfist_BTC(Stinkfist):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'inf-pct-adr': 0.91556,
        'mp': 66,
    }

    use_sell_signal = False

# Sub-strategy with parameters specific to ETH stake
class Stinkfist_ETH(Stinkfist):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'inf-pct-adr': 0.81628,
        'mp': 40,
    }

    trailing_stop = True
    trailing_stop_positive = 0.014
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = False

    use_sell_signal = False