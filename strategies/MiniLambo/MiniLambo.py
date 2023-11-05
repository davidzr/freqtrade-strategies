from skopt.space import Dimension, Integer

import logging
import logging
from datetime import datetime, timezone
from functools import reduce
from typing import List

import numpy as np
import pandas_ta as pta
import talib.abstract as ta
import technical.indicators as ftt
from pandas import DataFrame, Series
from skopt.space import Dimension, Integer

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, DecimalParameter,
                                IntParameter, merge_informative_pair)
from freqtrade.strategy.interface import IStrategy

logger = logging.getLogger(__name__)

# ###############################################################################
# ###############################################################################
# @Farhad#0318 ( https://github.com/farfary/freqtrade_strategies )
#
# Based on buy signal from Al (alb#1349)
# ###############################################################################
# ###############################################################################


class MiniLambo(IStrategy):
    # Protection hyperspace params:
    protection_params = {
        "protection_cooldown_period": 2,
        "protection_maxdrawdown_lookback_period_candles": 35,
        "protection_maxdrawdown_max_allowed_drawdown": 0.097,
        "protection_maxdrawdown_stop_duration_candles": 1,
        "protection_maxdrawdown_trade_limit": 6,
        "protection_stoplossguard_lookback_period_candles": 16,
        "protection_stoplossguard_stop_duration_candles": 29,
        "protection_stoplossguard_trade_limit": 3,
    }

    protection_cooldown_period = IntParameter(low=1, high=48, default=1, space="protection", optimize=True)

    protection_maxdrawdown_lookback_period_candles = IntParameter(low=1, high=48, default=1, space="protection", optimize=True)
    protection_maxdrawdown_trade_limit = IntParameter(low=1, high=8, default=4, space="protection", optimize=True)
    protection_maxdrawdown_stop_duration_candles = IntParameter(low=1, high=48, default=1, space="protection", optimize=True)
    protection_maxdrawdown_max_allowed_drawdown = DecimalParameter(low=0.01, high=0.20, default=0.1, space="protection", optimize=True)

    protection_stoplossguard_lookback_period_candles = IntParameter(low=1, high=48, default=1, space="protection", optimize=True)
    protection_stoplossguard_trade_limit = IntParameter(low=1, high=8, default=4, space="protection", optimize=True)
    protection_stoplossguard_stop_duration_candles = IntParameter(low=1, high=48, default=1, space="protection", optimize=True)

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.protection_cooldown_period.value
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": self.protection_maxdrawdown_lookback_period_candles.value,
                "trade_limit": self.protection_maxdrawdown_trade_limit.value,
                "stop_duration_candles": self.protection_maxdrawdown_stop_duration_candles.value,
                "max_allowed_drawdown": self.protection_maxdrawdown_max_allowed_drawdown.value
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.protection_stoplossguard_lookback_period_candles.value,
                "trade_limit": self.protection_stoplossguard_trade_limit.value,
                "stop_duration_candles": self.protection_stoplossguard_stop_duration_candles.value,
                "only_per_pair": False
            }
        ]

    class HyperOpt:
        @staticmethod
        def generate_roi_table(params: dict):
            """
            Generate the ROI table that will be used by Hyperopt
            This implementation generates the default legacy Freqtrade ROI tables.
            Change it if you need different number of steps in the generated
            ROI tables or other structure of the ROI tables.
            Please keep it aligned with parameters in the 'roi' optimization
            hyperspace defined by the roi_space method.
            """
            roi_table = {}
            roi_table[0] = 0.05
            roi_table[params['roi_t6']] = 0.04
            roi_table[params['roi_t5']] = 0.03
            roi_table[params['roi_t4']] = 0.02
            roi_table[params['roi_t3']] = 0.01
            roi_table[params['roi_t2']] = 0.0001
            roi_table[params['roi_t1']] = -10

            return roi_table

        @staticmethod
        def roi_space() -> List[Dimension]:
            """
            Values to search for each ROI steps
            Override it if you need some different ranges for the parameters in the
            'roi' optimization hyperspace.
            Please keep it aligned with the implementation of the
            generate_roi_table method.
            """
            return [
                Integer(240, 720, name='roi_t1'),
                Integer(120, 240, name='roi_t2'),
                Integer(90, 120, name='roi_t3'),
                Integer(60, 90, name='roi_t4'),
                Integer(30, 60, name='roi_t5'),
                Integer(1, 30, name='roi_t6'),
            ]

    # ROI table:
    minimal_roi = {
        "0": 0.05,
        "15": 0.04,
        "51": 0.03,
        "81": 0.02,
        "112": 0.01,
        "154": 0.0001,
        "400": -10
    }

    # Stoploss:
    stoploss = -0.10

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.3207
    trailing_stop_positive_offset = 0.3849
    trailing_only_offset_is_reached = False

    timeframe = '1m'

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 200

    plot_config = {
        "main_plot": {
            "ema_14": {
                "color": "#888b48",
                "type": "line"
            },
            "buy_sell": {
                "sell_tag": {"color": "red"},
                "buy_tag": {"color": "blue"},
            },
        },
        "subplots": {
            "rsi4": {
                "rsi_4": {
                    "color": "#888b48",
                    "type": "line"
                }
            },
            "rsi14": {
                "rsi_14": {
                    "color": "#888b48",
                    "type": "line"
                }
            },
            "cti": {
                "cti": {
                    "color": "#573892",
                    "type": "line"
                }
            },
            "ewo": {
                "EWO": {
                    "color": "#573892",
                    "type": "line"
                }
            },
            "pct_change": {
                "pct_change": {
                    "color": "#26782f",
                    "type": "line"
                }
            }
        }
    }

    # Buy hyperspace params:
    buy_params = {
        "lambo2_pct_change_high_period": 109,
        "lambo2_pct_change_high_ratio": -0.235,
        "lambo2_pct_change_low_period": 20,
        "lambo2_pct_change_low_ratio": -0.06,
        "lambo2_ema_14_factor": 0.981,
        "lambo2_rsi_14_limit": 39,
        "lambo2_rsi_21_limit": 39,
        "lambo2_rsi_4_limit": 44,
    }

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3, default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=False)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=False)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=False)
    lambo2_rsi_21_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_21_limit'], space='buy', optimize=False)

    lambo2_pct_change_low_period = IntParameter(1, 60, default=buy_params['lambo2_pct_change_low_period'], space='buy', optimize=True)
    lambo2_pct_change_low_ratio = DecimalParameter(low=-0.20, high=-0.01, decimals=3, default=buy_params['lambo2_pct_change_low_ratio'], space='buy', optimize=True)

    lambo2_pct_change_high_period = IntParameter(1, 180, default=buy_params['lambo2_pct_change_high_period'], space='buy', optimize=True)
    lambo2_pct_change_high_ratio = DecimalParameter(low=-0.30, high=-0.01, decimals=3, default=buy_params['lambo2_pct_change_high_ratio'], space='buy', optimize=True)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs += [("BTC/USDT", "1m")]
        informative_pairs += [("BTC/USDT", "1d")]

        return informative_pairs

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if (current_profit > 0.2):
            sl_new = 0.05
        elif (current_profit > 0.1):
            sl_new = 0.03
        elif (current_profit > 0.06):
            sl_new = 0.02
        elif (current_profit > 0.03):
            sl_new = 0.015
        elif (current_profit > 0.015):
            sl_new = 0.0075

        return sl_new

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_21'] = ta.RSI(dataframe, timeperiod=21)
        dataframe['rsi_100'] = ta.RSI(dataframe, timeperiod=100)

        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['ewo'] = EWO(dataframe, 50, 200)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        lambo2 = (
                (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value))
                & (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value))
                & (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))
                # & (dataframe['rsi_21'] > int(self.lambo2_rsi_21_limit.value))
                & (dataframe['close'].pct_change(periods=self.lambo2_pct_change_low_period.value) < float(self.lambo2_pct_change_low_ratio.value))
                & (dataframe['close'].pct_change(periods=self.lambo2_pct_change_high_period.value) > float(self.lambo2_pct_change_high_ratio.value))
        )
        dataframe.loc[lambo2, 'buy_tag'] += 'lambo2 '
        conditions.append(lambo2)

        dataframe.loc[reduce(lambda x, y: x | y, conditions), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        trade.sell_reason = f'{sell_reason} ({trade.buy_tag})'
        return True


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


def pct_change(a, b):
    return (b - a) / a


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


class MiniLambo_TBS(MiniLambo):
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
        if reinit or not 'trailing_buy' in self.custom_info_trail_buy[pair]:
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
                f"profit: {self.current_trailing_profit_ratio(pair, current_price) * 100:.2f}%, "
                f"offset: {trailing_buy['offset']}")

    def current_trailing_profit_ratio(self, pair: str, current_price: float) -> float:
        trailing_buy = self.trailing_buy(pair)
        if trailing_buy['trailing_buy_order_started']:
            return (trailing_buy['start_trailing_price'] - current_price) / trailing_buy['start_trailing_price']
        else:
            return 0

    def trailing_buy_offset(self, dataframe, pair: str, current_price: float):
        # return rebound limit before a buy in % of initial price, function of current price return None to stop
        # trailing buy (will start again at next buy signal) return 'forcebuy' to force immediate buy (example with
        # 0.5%. initial price : 100 (uplimit is 100.5), 2nd price : 99 (no buy, uplimit updated to 99.5), 3price 98 (
        # no buy uplimit updated to 98.5), 4th price 99 -> BUY
        current_trailing_profit_ratio = self.current_trailing_profit_ratio(pair, current_price)
        last_candle = dataframe.iloc[-1]
        adapt = abs((last_candle['perc_norm']))
        default_offset = 0.004 * (1 + adapt)  # NOTE: default_offset 0.003 <--> 0.006
        # default_offset = adapt*0.01

        trailing_buy = self.trailing_buy(pair)
        if not trailing_buy['trailing_buy_order_started']:
            return default_offset

        # example with duration and indicators
        # dry run, live only
        last_candle = dataframe.iloc[-1]
        current_time = datetime.now(timezone.utc)
        trailing_duration = current_time - trailing_buy['start_trailing_time']
        if trailing_duration.total_seconds() > self.trailing_expire_seconds:
            if (current_trailing_profit_ratio > 0) and (last_candle['buy'] == 1):
                # more than 1h, price under first signal, buy signal still active -> buy
                return 'forcebuy'
            else:
                # wait for next signal
                return None
        elif (self.trailing_buy_uptrend_enabled and (
                trailing_duration.total_seconds() < self.trailing_expire_seconds_uptrend) and (
                      current_trailing_profit_ratio < (-1 * self.min_uptrend_trailing_profit))):
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

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str,
                            **kwargs) -> bool:
        val = super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force, **kwargs)

        if val:
            if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):
                val = False
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if (len(dataframe) >= 1):
                    last_candle = dataframe.iloc[-1].squeeze()
                    current_price = rate
                    trailing_buy = self.trailing_buy(pair)
                    trailing_buy_offset = self.trailing_buy_offset(dataframe, pair, current_price)

                    if trailing_buy['allow_trailing']:
                        if not trailing_buy['trailing_buy_order_started'] and (last_candle['buy'] == 1):
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
                                logger.info(
                                    f"price OK for {pair} ({ratio} %, {current_price}), order may not be triggered if all slots are full")

                            elif trailing_buy_offset is None:
                                # stop trailing buy custom conditions
                                self.trailing_buy(pair, reinit=True)
                                logger.info(f'STOP trailing buy for {pair} because "trailing buy offset" returned None')

                            elif current_price < trailing_buy['trailing_buy_order_uplimit']:
                                # update uplimit
                                old_uplimit = trailing_buy["trailing_buy_order_uplimit"]
                                self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'] = min(
                                    current_price * (1 + trailing_buy_offset),
                                    self.custom_info_trail_buy[pair]['trailing_buy']['trailing_buy_order_uplimit'])
                                self.custom_info_trail_buy[pair]['trailing_buy']['offset'] = trailing_buy_offset
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f'update trailing buy for {pair} at {old_uplimit} -> {self.custom_info_trail_buy[pair]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                            elif current_price < (
                                    trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy)):
                                # buy ! current price > uplimit && lower thant starting price
                                val = True
                                ratio = "%.2f" % ((self.current_trailing_profit_ratio(pair, current_price)) * 100)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f"current price ({current_price}) > uplimit ({trailing_buy['trailing_buy_order_uplimit']}) and lower than starting price price ({(trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_buy))}). OK for {pair} ({ratio} %), order may not be triggered if all slots are full")

                            elif current_price > (
                                    trailing_buy['start_trailing_price'] * (1 + self.trailing_buy_max_stop)):
                                # stop trailing buy because price is too high
                                self.trailing_buy(pair, reinit=True)
                                self.trailing_buy_info(pair, current_price)
                                logger.info(
                                    f'STOP trailing buy for {pair} because of the price is higher than starting price * {1 + self.trailing_buy_max_stop}')
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
                    dataframe.loc[:, 'buy'] = 1
                    dataframe.loc[:, 'buy_tag'] = trailing_buy['buy_tag']
                    # dataframe['buy'] = 1

        return dataframe
