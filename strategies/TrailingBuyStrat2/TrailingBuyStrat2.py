# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
# --------------------------------

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
import time

logger = logging.getLogger(__name__)


class TrailingBuyStrat2(YourStrat):
    # Original idea by @MukavaValkku, code by @tirail and @stash86
    #
    # This class is designed to inherit from yours and starts trailing buy with your buy signals
    # Trailing buy starts at any buy signal and will move to next candles if the trailing still active
    # Trailing buy stops  with BUY if : price decreases and rises again more than trailing_buy_offset
    # Trailing buy stops with NO BUY : current price is > initial price * (1 +  trailing_buy_max) OR custom_sell tag
    # IT IS NOT COMPATIBLE WITH BACKTEST/HYPEROPT
    #

    process_only_new_candles = True

    custom_info_trail_buy = dict()

    # Trailing buy parameters
    trailing_buy_order_enabled = True
    trailing_expire_seconds = 1800

    # If the current candle goes above min_uptrend_trailing_profit % before trailing_expire_seconds_uptrend seconds, buy the coin
    trailing_buy_uptrend_enabled = False
    trailing_expire_seconds_uptrend = 90
    min_uptrend_trailing_profit = 0.02

    debug_mode = True
    trailing_buy_max_stop = 0.02  # stop trailing buy if current_price > starting_price * (1+trailing_buy_max_stop)
    trailing_buy_max_buy = 0.000  # buy if price between uplimit (=min of serie (current_price * (1 + trailing_buy_offset())) and (start_price * 1+trailing_buy_max_buy))

    init_trailing_dict = {
        'trailing_buy_order_started': False,
        'trailing_buy_order_uplimit': 0,
        'start_trailing_price': 0,
        'buy_tag': None,
        'start_trailing_time': None,
        'offset': 0,
        'allow_trailing': False,
    }

    def trailing_buy(self, pair, reinit=False):
        # returns trailing buy info for pair (init if necessary)
        if not pair in self.custom_info_trail_buy:
            self.custom_info_trail_buy[pair] = dict()
        if (reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]):
            self.custom_info_trail_buy[pair]['trailing_buy'] = self.init_trailing_dict.copy()
        return self.custom_info_trail_buy[pair]['trailing_buy']

    def trailing_buy_info(self, pair: str, current_price: float):
        # current_time live, dry run
        current_time = datetime.now(timezone.utc)
        if not self.debug_mode:
            return
        trailing_buy = self.trailing_buy(pair)

        duration = 0
        try:
            duration = (current_time - trailing_buy['start_trailing_time'])
        except TypeError:
            duration = 0
        finally:
            logger.info(
                f"pair: {pair} : "
                f"start: {trailing_buy['start_trailing_price']:.4f}, "
                f"duration: {duration}, "
                f"current: {current_price:.4f}, "
                f"uplimit: {trailing_buy['trailing_buy_order_uplimit']:.4f}, "
                f"profit: {self.current_trailing_profit_ratio(pair, current_price)*100:.2f}%, "
                f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price
        # return None to stop trailing buy (will start again at next buy signal)
        # return 'forcebuy' to force immediate buy
        # (example with 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        default_offset = 0.005

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if ((current_trailing_profit_ratio > 0) and (last_candle['buy'] == 1)):
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
            # less than 90s and price is rising, buy
            return 'forcebuy'

        if current_trailing_profit_ratio < 0:
            # current price is higher than initial price
            return default_offset

        trailing_buy_offset = {
            0.06: 0.02,
            0.03: 0.01,
            0: default_offset,
        }

        for key in trailing_buy_offset:
            if current_trailing_profit_ratio > key:
                return trailing_buy_offset[key]

        return default_offset

    # end of trailing buy parameters
    # -----------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        self.trailing_buy(metadata['pair'])
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)
        
        if val:
            if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if(len(dataframe) >= 1):
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_buy = self.trailing_buy(pair)
                    trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)

                    if trailing_buy['allow_trailing']:
                        if (not trailing_buy['trailing_buy_order_started'] and (last_candle['buy'] == 1)):
                            # start trailing buy
                            
                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_started'] = True
                            # self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_price'] = last_candle['close']
                            # self.custom_info_trail_buy[pair]['trailing_buy']['buy_tag'] = f"initial_buy_tag (strat trail price {last_candle['close']})"
                            # self.custom_info_trail_buy[pair]['trailing_buy']['start_trailing_time'] = datetime.now(timezone.utc)
                            # self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = 0

                            trailing_buy['trailing_buy_order_started'] = True
                            trailing_buy['trailing_buy_order_uplimit'] = last_candle['close']
                            trailing_buy['start_trailing_price'] = last_candle['close']
                            trailing_buy['buy_tag'] = last_candle['buy_tag']
                            trailing_buy['start_trailing_time'] = datetime.now(timezone.utc)
                            trailing_buy['offset'] = 0
                            
                            self.trailing_buy_info(pair, current_price)
                            logger.info(f'start trailing buy for {pair} at {last_candle["close"]}')

                        elif trailing_buy['trailing_buy_order_started']:
                            if trailing_buy_offset == 'forcebuy':
                                # buy in custom conditions
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

                            elif trailing_buy_offset is None:
                                # stop trailing buy custom conditions
                                self.trailing_buy(pair, reinit=True)
                                logger.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')

                            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                # update uplimit
                                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + trailing_buy_offset), self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                            elif current_price < (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                                # buy ! current price > uplimit && lower thant starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price price ({(trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy))}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")

                            elif current_price > (trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                                # stop trailing buy because price is too high
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
                            else:
                                # uplimit > current_price > max_price, continue trailing and wait for the price to go down
                                self.trailing_buy_info(pair, current_price)
                                logger.info(f'price too high for {pair} !')

                    else:
                        logger.info(f"Wait for next buy signal for {pair}")

                if (val == True):
                    self.trailing_buy_info(pair, rate)
                    self.trailing_buy(pair, reinit=True)
                    logger.info(f'STOP trailing buy for {pair} because I buy it')
        
        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'): 
            last_candle = dataframe.iloc[-1].squeeze()
            trailing_buy = self.trailing_buy(metadata['pair'])
            if (last_candle['buy'] == 1):
                if not trailing_buy['trailing_buy_order_started']:
                    open_trades = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True), ]).all()
                    if not open_trades:
                        logger.info(f"Set 'allow_trailing' to True for {metadata['pair']} to start trailing!!!")
                        # self.custom_info_trail_buy[metadata['pair']]['trailing_buy']['allow_trailing'] = True
                        trailing_buy['allow_trailing'] = True
                        initial_buy_tag = last_candle['buy_tag'] if 'buy_tag' in last_candle else 'buy signal'
                        dataframe.loc[:, 'buy_tag'] = f"{initial_buy_tag} (start trail price {last_candle['close']})"
            else:
                if (trailing_buy['trailing_buy_order_started'] == True):
                    logger.info(f"Continue trailing for {metadata['pair']}. Manually trigger buy signal!!")
                    dataframe.loc[:,'buy'] = 1
                    dataframe.loc[:, 'buy_tag'] = trailing_buy['buy_tag']
                    # dataframe['buy'] = 1

        return dataframe

