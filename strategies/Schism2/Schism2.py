import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from typing import Dict, List, Optional, Tuple
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI
from statistics import mean
from cachetools import TTLCache


"""
Solipsis - By @werkkrew and @JimmyNixx
This strategy is an evolution of our previous framework "Schism" which we are happy to share by request. 

FEATURES:
    - Sticking buy signal for extending ROI
        - Idea is to use a completely different buy signal for active trades to force the "ignore_roi_if_buy_signal = True" setting to stop a sell to ROI
          during a strong upward trend.
            - This is not compatible with backtest or hyperopt and can only be tested in dry-run or live.
    - Dynamic Sell
        - Emulates a pre-stoploss bailout when market conditions meet specific criteria, including looking at profit factors, other trade status, etc.
    - Dynamic informative indicators based on certain stake currences and whitelist contents.
        - If your stake is BTC or ETH, use COIN/FIAT and BTC/FIAT as informatives.
    - Full access to current trade and other trade data within buy and sell methods to use in decisions, this is not compatible with backtesting.
    - Ability to provide custom parameters on a per-pair or group of pairs basis, this includes buy/sell/minimal_roi/dynamic_roi/custom_stop settings, if one desired.
    - Stub Child strategies for stake specific settings and different settings for different instances.

TODO: 
    - Continue to hunt for a better all around buy signal.
    - Tweak ROI Ride
        - Maybe use free_slots as a factor in how eager we are to sell?
    - Tweak sell signal
        - Continue to evaluate good circumstances to sell vs hold
    - Figure out a way to directly feed a daily hyperopt output into a running strategy and reload it?
    - Per-pair automatic ROI based on ADR or something similar?
"""

class Schism2(IStrategy):
    """
    Strategy Configuration Items
    """
    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'inf-pct-adr': 0.83534,
        'inf-rsi': 57,
        'mp': 64,
        'rmi-fast': 49,
        'rmi-slow': 24,
        'xinf-stake-rmi': 45,
        'xtf-fiat-rsi': 28,
        'xtf-stake-rsi': 90
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

    custom_pair_params = []

    stoploss = -0.30

    # Recommended
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 72

    # Startegy Specific Variable Storage
    custom_trade_info = {}
    custom_fiat = "USD"
    custom_current_price_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # 5 minutes
    
    """
    Informative Pair Definitions
    """
    def informative_pairs(self):
        # add existing pairs from whitelist on the inf_timeframe
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        
        # add additional informative pairs based on certain stakes
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            for pair in pairs:
                # add in the COIN/FIAT pairs (e.g. XLM/USD) on base timeframe
                coin, stake = pair.split('/')
                coin_fiat = f"{coin}/{self.custom_fiat}"
                informative_pairs += [(coin_fiat, self.timeframe)]

            # add in the STAKE/FIAT pair (e.g. BTC/USD) on base and inf timeframes
            stake_fiat = f"{self.config['stake_currency']}/{self.custom_fiat}"
            informative_pairs += [(stake_fiat, self.timeframe)]
            informative_pairs += [(stake_fiat, self.inf_timeframe)]

        return informative_pairs

    """
    Indicator Definitions
    """ 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Populate/update the trade data if there is any, set trades to false if not live/dry
        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])
    
        # Set up primary indicators
        dataframe['rmi-slow'] = RMI(dataframe, length=21, mom=5)
        dataframe['rmi-fast'] = RMI(dataframe, length=8, mom=4)

        # Momentum Pinball
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=6)
        dataframe['mp']  = ta.RSI(dataframe['roc'], timeperiod=6)

        # Trend Calculations    
        dataframe['rmi-up'] = np.where(dataframe['rmi-slow'] >= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-dn'] = np.where(dataframe['rmi-slow'] <= dataframe['rmi-slow'].shift(),1,0)      
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(3, min_periods=1).sum() >= 2,1,0)      
        dataframe['rmi-dn-trend'] = np.where(dataframe['rmi-dn'].rolling(3, min_periods=1).sum() >= 2,1,0)

        # Informative for STAKE/FIAT and COIN/FIAT on default timeframe, only relevant if stake currency is BTC or ETH
        if self.config['stake_currency'] in ('BTC', 'ETH'):
            coin, stake = metadata['pair'].split('/')
            fiat = self.custom_fiat
            coin_fiat = f"{coin}/{fiat}"
            stake_fiat = f"{stake}/{fiat}"

            # COIN/FIAT (e.g. XLM/USD) - timeframe
            coin_fiat_tf = self.dp.get_pair_dataframe(pair=coin_fiat, timeframe=self.timeframe)
            dataframe[f"{fiat}_rsi"] = ta.RSI(coin_fiat_tf, timeperiod=14)

            # STAKE/FIAT (e.g. BTC/USD) - inf_timeframe
            stake_fiat_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.timeframe)
            stake_fiat_inf_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.inf_timeframe)

            dataframe[f"{stake}_rsi"] = ta.RSI(stake_fiat_tf, timeperiod=14)
            dataframe[f"{stake}_rmi_{self.inf_timeframe}"] = RMI(stake_fiat_inf_tf, length=21, mom=5)

        # Informative indicators for current pair on inf_timeframe
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['rsi'] = ta.RSI(informative, timeperiod=14)

        informative['1d_high'] = informative['close'].rolling(24).max()
        informative['3d_low'] = informative['close'].rolling(72).min()
        informative['adr'] = informative['1d_high'] - informative['3d_low']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    """
    Buy Trigger Signals
    """
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.get_pair_params(metadata['pair'], 'buy')
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # Persist a buy signal for existing trades to make use of ignore_roi_if_buy_signal = True
        # when this buy signal is not present a sell can happen according to the defined ROI table
        if trade_data['active_trade']:
            # peak_profit factor f(x)=1-x/400, rmi 30 -> 0.925, rmi 80 -> 0.80
            profit_factor = (1 - (dataframe['rmi-slow'].iloc[-1] / 400))
            # grow from 30 -> 70 after 720 minutes starting after 180 minutes
            rmi_grow = self.linear_growth(30, 70, 180, 720, trade_data['open_minutes'])

            conditions.append(dataframe['rmi-up-trend'] == 1)
            conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * profit_factor))
            conditions.append(dataframe['rmi-slow'] >= rmi_grow)

        # Normal buy triggers that apply to new trades we want to enter
        else:
            # Primary buy triggers
            conditions.append(
                # "buy the dip" based buy signal using momentum pinball and downward RMI
                (dataframe['close'] <= dataframe[f"3d_low_{self.inf_timeframe}"] + (params['inf-pct-adr'] * dataframe[f"adr_{self.inf_timeframe}"])) &
                (dataframe[f"rsi_{self.inf_timeframe}"] >= params['inf-rsi']) &
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['rmi-slow'] >= params['rmi-slow']) &
                (dataframe['rmi-fast'] <= params['rmi-fast']) &
                (dataframe['mp'] <= params['mp'])
            )

            # If the stake is BTC or ETH apply additional conditions
            if self.config['stake_currency'] in ('BTC', 'ETH'):
                # default timeframe conditions
                conditions.append(
                    (dataframe[f"{self.config['stake_currency']}_rsi"] < params['xtf-stake-rsi']) | 
                    (dataframe[f"{self.custom_fiat}_rsi"] > params['xtf-fiat-rsi'])
                )
                # informative timeframe conditions
                conditions.append(dataframe[f"{self.config['stake_currency']}_rmi_{self.inf_timeframe}"] < params['xinf-stake-rmi'])

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
        params = self.get_pair_params(metadata['pair'], 'sell')
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # In this strategy all sells for profit happen according to ROI
        # This sell signal is designed only as a "dynamic stoploss"

        # if we are in an active trade for this pair
        if trade_data['active_trade']:     
            # grow from -0.03 -> 0 after 300 minutes starting immediately
            loss_cutoff = self.linear_growth(-0.03, 0, 0, 300, trade_data['open_minutes'])

            # if we are at a loss, consider what the trend looks and preempt the stoploss
            conditions.append(
                (trade_data['current_profit'] < loss_cutoff) & 
                (trade_data['current_profit'] > self.stoploss) &  
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['volume'].gt(0))
            )
            
            # if the peak profit was positive at some point but never reached ROI, set a higher cross point for exit
            if trade_data['peak_profit'] > 0:
                conditions.append(qtpylib.crossed_below(dataframe['rmi-slow'], 50))
            # if the trade was always negative, the bounce we expected didn't happen
            else:
                conditions.append(qtpylib.crossed_below(dataframe['rmi-slow'], 10))

            # if there are other open trades in addition to this one, consider the average profit 
            # across them all and how many free slots we have in our sell decision
            if trade_data['other_trades']:
                if trade_data['free_slots'] > 0:
                    """
                    Less free slots, more willing to sell
                    1 / free_slots * x = 
                    1 slot = 1/1 * -0.04 = -0.04 -> only allow sells if avg_other_proift above -0.04
                    4 slot = 1/4 * -0.04 = -0.01 -> only allow sells is avg_other_profit above -0.01
                    """
                    max_market_down = -0.04 
                    hold_pct = (1/trade_data['free_slots']) * max_market_down
                    conditions.append(trade_data['avg_other_profit'] >= hold_pct)
                else:
                    # if were out of slots, allow the biggest losing trade to sell regardless of avg profit
                    conditions.append(trade_data['biggest_loser'] == True)

        # Impossible condition to satisfy the bot when it looks here and theres no active trade
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
    # Populate trades_data from the database
    def populate_trades(self, pair: str) -> dict:
        # Initialize the trades dict if it doesn't exist, persist it otherwise
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        # init the temp dicts and set the trade stuff to false
        trade_data = {}
        trade_data['active_trade'] = trade_data['other_trades'] = trade_data['biggest_loser'] = False

        # active trade stuff only works in live and dry, not backtest
        if self.config['runmode'].value in ('live', 'dry_run'):
            
            # find out if we have an open trade for this pair
            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True),]).all()

            # if so, get some information
            if active_trade:
                # get current price and update the min/max rate
                current_rate = self.get_current_price(pair, True)
                active_trade[0].adjust_min_max_rates(current_rate)

                # get how long the trade has been open in minutes and candles
                present = arrow.utcnow()
                trade_start  = arrow.get(active_trade[0].open_date)
                open_minutes = (present - trade_start).total_seconds() // 60  # floor

                # set up the things we use in the strategy
                trade_data['active_trade']   = True
                trade_data['current_profit'] = active_trade[0].calc_profit_ratio(current_rate)
                trade_data['peak_profit']    = max(0, active_trade[0].calc_profit_ratio(active_trade[0].max_rate))
                trade_data['open_minutes']   : int = open_minutes
                trade_data['open_candles']   : int = (open_minutes // active_trade[0].timeframe) # floor
            else: 
                trade_data['current_profit'] = trade_data['peak_profit']  = 0.0
                trade_data['open_minutes']   = trade_data['open_candles'] = 0

            # if there are open trades not including the current pair, get some information
            # future reference, for *all* open trades: open_trades = Trade.get_open_trades()
            other_trades = Trade.get_trades([Trade.pair != pair, Trade.is_open.is_(True),]).all()

            if other_trades:
                trade_data['other_trades'] = True
                other_profit = tuple(trade.calc_profit_ratio(self.get_current_price(trade.pair, False)) for trade in other_trades)
                trade_data['avg_other_profit'] = mean(other_profit) 
                # find which of our trades is the biggest loser
                if trade_data['current_profit'] < min(other_profit):
                    trade_data['biggest_loser'] = True
            else:
                trade_data['avg_other_profit'] = 0

            # get the number of free trade slots, storing in every pairs dict due to laziness
            open_trades = len(Trade.get_open_trades())
            trade_data['free_slots'] = max(0, self.config['max_open_trades'] - open_trades)

        return trade_data

    # Get the current price from the exchange (or cache)
    def get_current_price(self, pair: str, refresh: bool) -> float:
        if not refresh:
            rate = self.custom_current_price_cache.get(pair)
            # Check if cache has been invalidated
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

    # linear growth, starts at X and grows to Y after A minutes (starting after B miniutes)
    # f(t) = X + (rate * t), where rate = (Y - X) / (A - B)
    def linear_growth(self, start: float, end: float, start_time: int, end_time: int, trade_time: int) -> float:
        time = max(0, trade_time - start_time)
        rate = (end - start) / (end_time - start_time)
        return min(end, start + (rate * time))

    """
    Allow for buy/sell override parameters per pair. Testing, might remove.
    """
    def get_pair_params(self, pair: str, params: str) -> Dict:
        buy_params = self.buy_params
        sell_params = self.sell_params
        minimal_roi = self.minimal_roi
  
        if self.custom_pair_params:
            custom_params = next(item for item in self.custom_pair_params if pair in item['pairs'])
            if custom_params['buy_params']:
                buy_params = custom_params['buy_params']
            if custom_params['sell_params']:
                sell_params = custom_params['sell_params']
            if custom_params['minimal_roi']:
                custom_stop = custom_params['minimal_roi']
            
        if params == 'buy':
            return buy_params
        if params == 'sell':
            return sell_params
        if params == 'minimal_roi':
            return minimal_roi

    """
    Price protection on trade entry and timeouts, built-in Freqtrade functionality
    https://www.freqtrade.io/en/latest/strategy-advanced/
    """
    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > order['price'] * 1.01:
            return True
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ask_strategy = self.config.get('ask_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{ask_strategy['price_side']}s"][0][0]
        # Cancel sell order if price is more than 1% below the order.
        if current_price < order['price'] * 0.99:
            return True
        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        bid_strategy = self.config.get('bid_strategy', {})
        ob = self.dp.orderbook(pair, 1)
        current_price = ob[f"{bid_strategy['price_side']}s"][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > rate * 1.01:
            return False
        return True

    """
    ROI overrides for per-pair params
    """
    def min_roi_reached_entry(self, trade_dur: int, pair: str = 'backtest') -> Tuple[Optional[int], Optional[float]]:
        minimal_roi = self.get_pair_params(pair, 'minimal_roi')

        roi_list = list(filter(lambda x: x <= trade_dur, minimal_roi.keys()))
        if not roi_list:
            return None, None
        roi_entry = max(roi_list)

        return roi_entry, minimal_roi[roi_entry]

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:  
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        _, roi = self.min_roi_reached_entry(trade_dur, trade.pair)
        if roi is None:
            return False
        else:
            return current_profit > roi

"""
Sub-strategy overrides
Anything not explicity defined here will follow the settings in the base strategy
"""
# Sub-strategy with parameters specific to BTC stake
class Schism2_BTC(Schism2):

    timeframe = '15m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.05,
        "30": 0.025,
        "60": 0.015,
        "90": 0.01,
        "1440": 0.005,
        "2880": 0
    }

    buy_params = {
        'inf-pct-adr': 0.80616,
        'inf-rsi': 14,
        'mp': 43,
        'rmi-fast': 33,
        'rmi-slow': 16,
        'xinf-stake-rmi': 29,
        'xtf-fiat-rsi': 49,
        'xtf-stake-rsi': 53
    }

    use_sell_signal = False

# Sub-strategy with parameters specific to ETH stake
class Schism2_ETH(Schism2):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'inf-pct-adr': 0.81628,
        'inf-rsi': 13,
        'xinf-stake-rmi': 69,
        'mp': 40,
        'rmi-fast': 42,
        'rmi-slow': 17,
        'xtf-fiat-rsi': 15,
        'xtf-stake-rsi': 92
    }

    use_sell_signal = False