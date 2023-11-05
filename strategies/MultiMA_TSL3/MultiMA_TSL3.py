import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, BooleanParameter, timeframe_to_minutes, stoploss_from_open)
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_prev_date
from technical.indicators import zema, VIDYA


###########################################################################################################
##    MultiMA_TSL, modded by stash86, based on SMAOffsetProtectOptV1 (modded by Perkmeister)             ##
##    Based on @Lamborghini Store's SMAOffsetProtect strat, heavily based on @tirail's original SMAOffset##
##                                                                                                       ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##
##                                                                                                       ##
##    Thanks to                                                                                          ##
##    - Perkmeister, for their snippets for the sell signals and decaying EMA sell                       ##
##    - ChangeToTower, for the PMax idea                                                                 ##
##    - JimmyNixx, for their snippet to limit close value from the peak (that I modify into 5m tf check) ##
##    - froggleston, for the Heikinashi check snippet from Cryptofrog                                    ##
##    - Uzirox, for their pump detection code                                                            ##
##                                                                                                       ##
##                                                                                                       ##
###########################################################################################################

# I hope you do enough testing before proceeding, either backtesting and/or dry run.
# Any profits and losses are all your responsibility

class MultiMA_TSL3(IStrategy):
    INTERFACE_VERSION = 2

    DATESTAMP = 0
    SELLMA = 1
    SELL_TRIGGER = 2

    buy_params = {
        "base_nb_candles_buy_trima": 15,
        "base_nb_candles_buy_trima2": 38,
        "low_offset_trima": 0.959,
        "low_offset_trima2": 0.949,

        "base_nb_candles_buy_ema": 9,
        "base_nb_candles_buy_ema2": 75,
        "low_offset_ema": 1.067,
        "low_offset_ema2": 0.973,

        "base_nb_candles_buy_zema": 25,
        "base_nb_candles_buy_zema2": 53,
        "low_offset_zema": 0.958,
        "low_offset_zema2": 0.961,

        "base_nb_candles_buy_hma": 70,
        "base_nb_candles_buy_hma2": 12,
        "low_offset_hma": 0.948,
        "low_offset_hma2": 0.941,

        "ewo_high": 2.615,
        "ewo_high2": 2.188,
        "ewo_low": -19.632,
        "ewo_low2": -19.955,
        "rsi_buy": 60,
        "rsi_buy2": 45,

    }

    sell_params = {
        "base_nb_candles_ema_sell": 5,
        "high_offset_sell_ema": 0.994,
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.32,
        "pPF_1": 0.02,
        "pPF_2": 0.047,
        "pSL_1": 0.02,
        "pSL_2": 0.046,
    }

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    stoploss = -0.15

    # hard stoploss profit
    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)

    optimize_sell_ema = True
    base_nb_candles_ema_sell = IntParameter(5, 80, default=20, space='sell', optimize=True)
    high_offset_sell_ema = DecimalParameter(0.99, 1.1, default=1.012, space='sell', optimize=True)
    base_nb_candles_ema_sell2 = IntParameter(5, 80, default=20, space='sell', optimize=True)

    # Multi Offset
    optimize_buy_ema = True
    base_nb_candles_buy_ema = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_ema)
    low_offset_ema = DecimalParameter(0.9, 1.1, default=0.958, space='buy', optimize=optimize_buy_ema)
    base_nb_candles_buy_ema2 = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_ema)
    low_offset_ema2 = DecimalParameter(0.9, 1.1, default=0.958, space='buy', optimize=optimize_buy_ema)

    optimize_buy_trima = True
    base_nb_candles_buy_trima = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_trima)
    low_offset_trima = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_trima)
    base_nb_candles_buy_trima2 = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_trima)
    low_offset_trima2 = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_trima)

    optimize_buy_zema = True
    base_nb_candles_buy_zema = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_zema)
    low_offset_zema = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_zema)
    base_nb_candles_buy_zema2 = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_zema)
    low_offset_zema2 = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_zema)

    optimize_buy_hma = True
    base_nb_candles_buy_hma = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_hma)
    low_offset_hma = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_hma)
    base_nb_candles_buy_hma2 = IntParameter(5, 80, default=20, space='buy', optimize=optimize_buy_hma)
    low_offset_hma2 = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=optimize_buy_hma)

    buy_condition_enable_optimize = True
    buy_condition_trima_enable = BooleanParameter(default=True, space='buy', optimize=buy_condition_enable_optimize)
    buy_condition_zema_enable = BooleanParameter(default=True, space='buy', optimize=buy_condition_enable_optimize)
    buy_condition_hma_enable = BooleanParameter(default=True, space='buy', optimize=buy_condition_enable_optimize)

    # Protection
    ewo_check_optimize = True
    ewo_low = DecimalParameter(-20.0, -8.0, default=-20.0, space='buy', optimize=ewo_check_optimize)
    ewo_high = DecimalParameter(2.0, 12.0, default=6.0, space='buy', optimize=ewo_check_optimize)
    ewo_low2 = DecimalParameter(-20.0, -8.0, default=-20.0, space='buy', optimize=ewo_check_optimize)
    ewo_high2 = DecimalParameter(2.0, 12.0, default=6.0, space='buy', optimize=ewo_check_optimize)

    rsi_buy_optimize = True
    rsi_buy = IntParameter(30, 70, default=50, space='buy', optimize=rsi_buy_optimize)
    rsi_buy2 = IntParameter(30, 70, default=50, space='buy', optimize=rsi_buy_optimize)
    buy_rsi_fast = IntParameter(0, 50, default=35, space='buy', optimize=True)

    fast_ewo = IntParameter(10, 50, default=50, space='buy', optimize=True)
    slow_ewo = IntParameter(100, 200, default=200, space='buy', optimize=True)

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.018

    use_custom_stoploss = True

    # Protection hyperspace params:
    protection_params = {
        "low_profit_lookback": 48,
        "low_profit_min_req": 0.04,
        "low_profit_stop_duration": 14,

        "cooldown_lookback": 2,  # value loaded from strategy
        "stoploss_lookback": 72,  # value loaded from strategy
        "stoploss_stop_duration": 20,  # value loaded from strategy
    }

    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=True)

    low_profit_optimize = True
    low_profit_lookback = IntParameter(2, 60, default=20, space="protection", optimize=low_profit_optimize)
    low_profit_stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=low_profit_optimize)
    low_profit_min_req = DecimalParameter(-0.05, 0.05, default=-0.05, space="protection", decimals=2,
                                          optimize=low_profit_optimize)

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        prot.append({
            "method": "LowProfitPairs",
            "lookback_period_candles": self.low_profit_lookback.value,
            "trade_limit": 1,
            "stop_duration": int(self.low_profit_stop_duration.value),
            "required_profit": self.low_profit_min_req.value
        })

        return prot

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # storage dict for custom info
    custom_info = {}

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 400

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if (len(dataframe) < 1):
            return False
        last_candle = dataframe.iloc[-1]

        if (self.custom_info[pair][self.DATESTAMP] != last_candle['date']):
            # new candle, update EMA and check sell

            # smoothing coefficients
            sell_ema = self.custom_info[pair][self.SELLMA]
            if (sell_ema == 0):
                sell_ema = last_candle['ema_sell']
            emaLength = 32
            alpha = 2 / (1 + emaLength)

            # update sell_ema
            sell_ema = (alpha * last_candle['close']) + ((1 - alpha) * sell_ema)
            self.custom_info[pair][self.SELLMA] = sell_ema
            self.custom_info[pair][self.DATESTAMP] = last_candle['date']

            if ((last_candle['close'] > (sell_ema * self.high_offset_sell_ema.value)) & (last_candle['buy_copy'] == 0)):
                if self.config['runmode'].value in ('live', 'dry_run'):
                    self.custom_info[pair][self.SELL_TRIGGER] = 1
                    return False

                buy_tag = 'empty'

                if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
                    buy_tag = trade.buy_tag
                else:
                    trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
                    buy_signal = dataframe.loc[dataframe['date'] < trade_open_date]
                    if not buy_signal.empty:
                        buy_signal_candle = buy_signal.iloc[-1]
                        buy_tag = buy_signal_candle['buy_tag'] if buy_signal_candle['buy_tag'] != '' else 'empty'

                return f'New Sell Signal ({buy_tag})'

        return False

    # credit to Perkmeister for this custom stoploss to help the strategy ride a green candle when the sell signal triggered
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str,
                            **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if (len(dataframe) < 1):
            return False
        last_candle = dataframe.iloc[-1].squeeze()
        if ((rate > last_candle['close'])):
            return False

        self.custom_info[pair][self.DATESTAMP] = last_candle['date']
        self.custom_info[pair][self.SELLMA] = last_candle['ema_sell']

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        self.custom_info[pair][self.SELL_TRIGGER] = 0
        return True

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EWO
        dataframe['ewo'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        heikinashi["volume"] = dataframe["volume"]

        # Profit Maximizer - PMAX
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close']) / 4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)

        dataframe = HA(dataframe, 4)

        if self.config['runmode'].value in ('live', 'dry_run'):
            # Exchange downtime protection
            dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)
        else:
            dataframe['live_data_ok'] = True

        # Check if the entry already exists
        if not metadata["pair"] in self.custom_info:
            # Create empty entry for this pair {datestamp, sellma, sell_trigger}
            self.custom_info[metadata["pair"]] = ['', 0, 0]

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe['ema_offset_buy'] = ta.EMA(dataframe,
                                             int(self.base_nb_candles_buy_ema.value)) * self.low_offset_ema.value
        dataframe['ema_offset_buy2'] = ta.EMA(dataframe,
                                              int(self.base_nb_candles_buy_ema2.value)) * self.low_offset_ema2.value
        dataframe['ema_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_ema_sell.value))

        dataframe.loc[:, 'buy_tag'] = ''
        dataframe.loc[:, 'buy_copy'] = 0
        dataframe.loc[:, 'buy'] = 0

        if (self.buy_condition_trima_enable.value):
            dataframe['trima_offset_buy'] = ta.TRIMA(dataframe,
                                                     int(self.base_nb_candles_buy_trima.value)) * self.low_offset_trima.value
            dataframe['trima_offset_buy2'] = ta.TRIMA(dataframe,
                                                      int(self.base_nb_candles_buy_trima2.value)) * self.low_offset_trima2.value

            buy_offset_trima = (
                    (
                            (dataframe['close'] < dataframe['trima_offset_buy'])
                            &
                            (dataframe['pm'] <= dataframe['pmax_thresh'])
                    )
                    |
                    (
                            (dataframe['close'] < dataframe['trima_offset_buy2'])
                            &
                            (dataframe['pm'] > dataframe['pmax_thresh'])
                    )
            )
            dataframe.loc[buy_offset_trima, 'buy_tag'] += 'trima '
            conditions.append(buy_offset_trima)

        if (self.buy_condition_zema_enable.value):
            dataframe['zema_offset_buy'] = zema(dataframe,
                                                int(self.base_nb_candles_buy_zema.value)) * self.low_offset_zema.value
            dataframe['zema_offset_buy2'] = zema(dataframe,
                                                 int(self.base_nb_candles_buy_zema2.value)) * self.low_offset_zema2.value
            buy_offset_zema = (
                    (
                            (dataframe['close'] < dataframe['zema_offset_buy'])
                            &
                            (dataframe['pm'] <= dataframe['pmax_thresh'])
                    )
                    |
                    (
                            (dataframe['close'] < dataframe['zema_offset_buy2'])
                            &
                            (dataframe['pm'] > dataframe['pmax_thresh'])
                    )
            )
            dataframe.loc[buy_offset_zema, 'buy_tag'] += 'zema '
            conditions.append(buy_offset_zema)

        if (self.buy_condition_hma_enable.value):
            dataframe['hma_offset_buy'] = qtpylib.hull_moving_average(dataframe['close'], window=int(
                self.base_nb_candles_buy_hma.value)) * self.low_offset_hma.value
            dataframe['hma_offset_buy2'] = qtpylib.hull_moving_average(dataframe['close'], window=int(
                self.base_nb_candles_buy_hma2.value)) * self.low_offset_hma2.value
            buy_offset_hma = (
                    (
                            (
                                    (dataframe['close'] < dataframe['hma_offset_buy'])
                                    &
                                    (dataframe['pm'] <= dataframe['pmax_thresh'])
                                    &
                                    (dataframe['rsi'] < 35)

                            )
                            |
                            (
                                    (dataframe['close'] < dataframe['hma_offset_buy2'])
                                    &
                                    (dataframe['pm'] > dataframe['pmax_thresh'])
                                    &
                                    (dataframe['rsi'] < 30)
                            )
                    )
                    &
                    (dataframe['rsi_fast'] < 30)

            )
            dataframe.loc[buy_offset_hma, 'buy_tag'] += 'hma '
            conditions.append(buy_offset_hma)

        add_check = (
                (dataframe['live_data_ok'])
                &
                (dataframe['close'] < dataframe['Smooth_HA_L'])
                &
                (dataframe['close'] < (dataframe['ema_sell'] * self.high_offset_sell_ema.value))
                &
                (dataframe['close'].rolling(288).max() >= (dataframe['close'] * 1.10))
                &
                (dataframe['Smooth_HA_O'].shift(1) < dataframe['Smooth_HA_H'].shift(1))
                &
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value)
                &
                (dataframe['rsi_84'] < 60)
                &
                (dataframe['rsi_112'] < 60)
                &
                (
                        (
                                (dataframe['close'] < dataframe['ema_offset_buy'])
                                &
                                (dataframe['pm'] <= dataframe['pmax_thresh'])
                                &
                                (
                                        (dataframe['ewo'] < self.ewo_low.value)
                                        |
                                        (
                                                (dataframe['ewo'] > self.ewo_high.value)
                                                &
                                                (dataframe['rsi'] < self.rsi_buy.value)
                                        )
                                )
                        )
                        |
                        (
                                (dataframe['close'] < dataframe['ema_offset_buy2'])
                                &
                                (dataframe['pm'] > dataframe['pmax_thresh'])
                                &
                                (
                                        (dataframe['ewo'] < self.ewo_low2.value)
                                        |
                                        (
                                                (dataframe['ewo'] > self.ewo_high2.value)
                                                &
                                                (dataframe['rsi'] < self.rsi_buy2.value)
                                        )
                                )
                        )
                )
                &
                (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                (add_check & reduce(lambda x, y: x | y, conditions)),
                ['buy_copy', 'buy']
            ] = (1, 1)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = 0

        return dataframe


class MultiMA_TSL3a(MultiMA_TSL3):
    informative_timeframe = '1h'
    timeframe_15m = '15m'

    min_rsi_sell = 50
    min_rsi_sell_15m = 70

    max_change_pump = 35

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        informative_pairs.extend([(pair, self.timeframe_15m) for pair in pairs])
        return informative_pairs

    def get_informative_15m_indicators(self, metadata: dict):
        dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.timeframe_15m)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_15m = self.get_informative_15m_indicators(metadata)
        dataframe = merge_informative_pair(dataframe, informative_15m, self.timeframe, self.timeframe_15m, ffill=True)
        drop_columns = [(s + "_" + self.timeframe_15m) for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        # EWO
        dataframe['ewo'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # Heiken Ashi
        heikinashi = qtpylib.heikinashi(dataframe)
        heikinashi["volume"] = dataframe["volume"]

        # Profit Maximizer - PMAX
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close']) / 4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)

        dataframe = HA(dataframe, 4)

        # pump detector
        dataframe['pump'] = pump_warning(dataframe, perc=int(self.max_change_pump))  # 25% di pump

        if self.config['runmode'].value in ('live', 'dry_run'):
            # Exchange downtime protection
            dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)
        else:
            dataframe['live_data_ok'] = True

        # Check if the entry already exists
        if not metadata["pair"] in self.custom_info:
            # Create empty entry for this pair {datestamp, sellma, sell_trigger}
            self.custom_info[metadata["pair"]] = ['', 0, 0]

        return dataframe

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if (len(dataframe) < 1):
            return False
        last_candle = dataframe.iloc[-1]

        if (self.custom_info[pair][self.DATESTAMP] != last_candle['date']):
            # new candle, update EMA and check sell

            # smoothing coefficients
            sell_ema = self.custom_info[pair][self.SELLMA]
            if (sell_ema == 0):
                sell_ema = last_candle['ema_sell']
            emaLength = 32
            alpha = 2 / (1 + emaLength)

            # update sell_ema
            sell_ema = (alpha * last_candle['close']) + ((1 - alpha) * sell_ema)
            self.custom_info[pair][self.SELLMA] = sell_ema
            self.custom_info[pair][self.DATESTAMP] = last_candle['date']

            sell_tag = ''

            if (
                    (last_candle['close'] > (sell_ema * self.high_offset_sell_ema.value))
                    &
                    (last_candle['buy_copy'] == 0)
                    &
                    (last_candle['rsi'] > self.min_rsi_sell)
            ):
                if self.config['runmode'].value in ('live', 'dry_run'):
                    self.custom_info[pair][self.SELL_TRIGGER] = 1
                    return False

                sell_tag = 'Decaying EMA'

            if (
                    (last_candle['rsi_fast_15m'] > self.min_rsi_sell_15m)
                    &
                    (last_candle['buy_copy'] == 0)
                    &
                    (last_candle['rsi'] > self.min_rsi_sell)
            ):
                if self.config['runmode'].value in ('live', 'dry_run'):
                    self.custom_info[pair][self.SELL_TRIGGER] = 1
                    return False

                sell_tag = 'RSI 15m Overbought'

            if not (sell_tag == ''):
                buy_tag = 'empty'

                if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
                    buy_tag = trade.buy_tag
                else:
                    trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
                    buy_signal = dataframe.loc[dataframe['date'] < trade_open_date]
                    if not buy_signal.empty:
                        buy_signal_candle = buy_signal.iloc[-1]
                        buy_tag = buy_signal_candle['buy_tag'] if buy_signal_candle['buy_tag'] != '' else 'empty'

                return f'{sell_tag} ({buy_tag})'

        return False

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe['ema_offset_buy'] = ta.EMA(dataframe,
                                             int(self.base_nb_candles_buy_ema.value)) * self.low_offset_ema.value
        dataframe['ema_offset_buy2'] = ta.EMA(dataframe,
                                              int(self.base_nb_candles_buy_ema2.value)) * self.low_offset_ema2.value
        dataframe['ema_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_ema_sell.value))

        dataframe.loc[:, 'buy_tag'] = ''
        dataframe.loc[:, 'buy_copy'] = 0
        dataframe.loc[:, 'buy'] = 0

        if (self.buy_condition_trima_enable.value):
            dataframe['trima_offset_buy'] = ta.TRIMA(dataframe,
                                                     int(self.base_nb_candles_buy_trima.value)) * self.low_offset_trima.value
            dataframe['trima_offset_buy2'] = ta.TRIMA(dataframe,
                                                      int(self.base_nb_candles_buy_trima2.value)) * self.low_offset_trima2.value

            buy_offset_trima = (
                    (
                            (dataframe['close'] < dataframe['trima_offset_buy'])
                            &
                            (dataframe['pm'] <= dataframe['pmax_thresh'])
                    )
                    |
                    (
                            (dataframe['close'] < dataframe['trima_offset_buy2'])
                            &
                            (dataframe['pm'] > dataframe['pmax_thresh'])
                    )
            )
            dataframe.loc[buy_offset_trima, 'buy_tag'] += 'trima '
            conditions.append(buy_offset_trima)

        if (self.buy_condition_zema_enable.value):
            dataframe['zema_offset_buy'] = zema(dataframe,
                                                int(self.base_nb_candles_buy_zema.value)) * self.low_offset_zema.value
            dataframe['zema_offset_buy2'] = zema(dataframe,
                                                 int(self.base_nb_candles_buy_zema2.value)) * self.low_offset_zema2.value
            buy_offset_zema = (
                    (
                            (dataframe['close'] < dataframe['zema_offset_buy'])
                            &
                            (dataframe['pm'] <= dataframe['pmax_thresh'])
                    )
                    |
                    (
                            (dataframe['close'] < dataframe['zema_offset_buy2'])
                            &
                            (dataframe['pm'] > dataframe['pmax_thresh'])
                    )
            )
            dataframe.loc[buy_offset_zema, 'buy_tag'] += 'zema '
            conditions.append(buy_offset_zema)

        if (self.buy_condition_hma_enable.value):
            dataframe['hma_offset_buy'] = qtpylib.hull_moving_average(dataframe['close'], window=int(
                self.base_nb_candles_buy_hma.value)) * self.low_offset_hma.value
            dataframe['hma_offset_buy2'] = qtpylib.hull_moving_average(dataframe['close'], window=int(
                self.base_nb_candles_buy_hma2.value)) * self.low_offset_hma2.value
            buy_offset_hma = (
                    (
                            (
                                    (dataframe['close'] < dataframe['hma_offset_buy'])
                                    &
                                    (dataframe['pm'] <= dataframe['pmax_thresh'])
                                    &
                                    (dataframe['rsi'] < 35)

                            )
                            |
                            (
                                    (dataframe['close'] < dataframe['hma_offset_buy2'])
                                    &
                                    (dataframe['pm'] > dataframe['pmax_thresh'])
                                    &
                                    (dataframe['rsi'] < 30)
                            )
                    )
                    &
                    (dataframe['rsi_fast'] < 30)

            )
            dataframe.loc[buy_offset_hma, 'buy_tag'] += 'hma '
            conditions.append(buy_offset_hma)

        add_check = (
                (dataframe['live_data_ok'])
                &
                (dataframe['close'] < dataframe['Smooth_HA_L'])
                &
                (dataframe['close'] < (dataframe['ema_sell'] * self.high_offset_sell_ema.value))
                &
                (dataframe['close'].rolling(288).max() >= (dataframe['close'] * 1.10))
                &
                (dataframe['Smooth_HA_O'].shift(1) < dataframe['Smooth_HA_H'].shift(1))
                &
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value)
                &
                (dataframe['rsi_84'] < 60)
                &
                (dataframe['rsi_112'] < 60)
                &
                (dataframe['pump'].rolling(20).max() < 1)
                &
                (
                        (
                                (dataframe['close'] < dataframe['ema_offset_buy'])
                                &
                                (dataframe['pm'] <= dataframe['pmax_thresh'])
                                &
                                (
                                        (dataframe['ewo'] < self.ewo_low.value)
                                        |
                                        (
                                                (dataframe['ewo'] > self.ewo_high.value)
                                                &
                                                (dataframe['rsi'] < self.rsi_buy.value)
                                        )
                                )
                        )
                        |
                        (
                                (dataframe['close'] < dataframe['ema_offset_buy2'])
                                &
                                (dataframe['pm'] > dataframe['pmax_thresh'])
                                &
                                (
                                        (dataframe['ewo'] < self.ewo_low2.value)
                                        |
                                        (
                                                (dataframe['ewo'] > self.ewo_high2.value)
                                                &
                                                (dataframe['rsi'] < self.rsi_buy2.value)
                                        )
                                )
                        )
                )
                &
                (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                (add_check & reduce(lambda x, y: x | y, conditions)),
                ['buy_copy', 'buy']
            ] = (1, 1)

        return dataframe


# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


# PMAX
def pmax(df, period, multiplier, length, MAtype, src):
    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier / 10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier / 10) * df[atr])

    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
                basic_ub[i] < final_ub[i - 1]
                or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
                basic_lb[i] > final_lb[i - 1]
                or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                            and mavalue[i] <= final_ub[i])
            else final_lb[i] if (
                    pm_arr[i - 1] == final_ub[i - 1]
                    and mavalue[i] > final_ub[i]) else final_lb[i]
            if (pm_arr[i - 1] == final_lb[i - 1]
                and mavalue[i] >= final_lb[i]) else final_ub[i]
            if (pm_arr[i - 1] == final_lb[i - 1]
                and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down', 'up'), np.NaN)

    return pm, pmx


# smoothed Heiken Ashi
def HA(dataframe, smoothing=None):
    df = dataframe.copy()

    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    df.reset_index(inplace=True)

    ha_open = [(df['open'][0] + df['close'][0]) / 2]
    [ha_open.append((ha_open[i] + df['HA_Close'].values[i]) / 2) for i in range(0, len(df) - 1)]
    df['HA_Open'] = ha_open

    df.set_index('index', inplace=True)

    df['HA_High'] = df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
    df['HA_Low'] = df[['HA_Open', 'HA_Close', 'low']].min(axis=1)

    if smoothing is not None:
        sml = abs(int(smoothing))
        if sml > 0:
            df['Smooth_HA_O'] = ta.EMA(df['HA_Open'], sml)
            df['Smooth_HA_C'] = ta.EMA(df['HA_Close'], sml)
            df['Smooth_HA_H'] = ta.EMA(df['HA_High'], sml)
            df['Smooth_HA_L'] = ta.EMA(df['HA_Low'], sml)

    return df


def pump_warning(dataframe, perc=15):
    # NOTE: segna "1" se c'è un pump
    df = dataframe.copy()
    df["change"] = df["high"] - df["low"]
    df["test1"] = (df["close"] > df["open"])
    df["test2"] = ((df["change"] / df["low"]) > (perc / 100))
    df["result"] = (df["test1"] & df["test2"]).astype('int')
    return df['result']