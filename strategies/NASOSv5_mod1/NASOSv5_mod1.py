# --- Do not remove these libs ---
# --- Do not remove these libs ---
from logging import FATAL
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
import logging
import pandas as pd


logger = logging.getLogger(__name__)

# @Rallipanos
# @pluxury
# with help from @stash86 and @Perkmeister



def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


class NASOSv5_mod1(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 20,
        "ewo_high": 4.299,
        "ewo_high_2": 8.492,
        "ewo_low": -8.476,
        "low_offset": 0.984,
        "low_offset_2": 0.901,
        "lookback_candles": 7,
        "profit_threshold": 1.036,
        "rsi_buy": 80,
        "rsi_fast_buy": 27,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 20,
        "high_offset": 1.01,
        "high_offset_2": 1.142,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.4
    }

    # Stoploss:
    stoploss = -0.3  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.001  # value loaded from strategy
    trailing_stop_positive_offset = 0.03  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy



    # SMAOffset
    base_nb_candles_buy = IntParameter(
        2, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        2, 25, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    lookback_candles = IntParameter(
        1, 36, default=buy_params['lookback_candles'], space='buy', optimize=True)

    profit_threshold = DecimalParameter(0.99, 1.05,
                                        default=buy_params['profit_threshold'], space='buy', optimize=True)

    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=True)

    rsi_buy = IntParameter(10, 80, default=buy_params['rsi_buy'], space='buy', optimize=True)
    rsi_fast_buy = IntParameter(
        10, 50, default=buy_params['rsi_fast_buy'], space='buy', optimize=True)


    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_15m = '15m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200
    use_custom_stoploss = False

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
        'subplots': {
            'rsi': {
                'rsi': {'color': 'orange'},
                'rsi_fast': {'color': 'red'},
                'rsi_slow': {'color': 'green'},
            },
            'ewo': {
                'EWO': {'color': 'orange'}
            },
        }
    }

    slippage_protection = {
        'retries': 3,
        'max_slippage': -0.02
    }

    protections = [
        # 	{
        # 		"method": "StoplossGuard",
        # 		"lookback_period_candles": 12,
        # 		"trade_limit": 1,
        # 		"stop_duration_candles": 6,
        # 		"only_per_pair": True
        # 	},
        # 	{
        # 		"method": "StoplossGuard",
        # 		"lookback_period_candles": 12,
        # 		"trade_limit": 2,
        # 		"stop_duration_candles": 6,
        # 		"only_per_pair": False
        # 	},
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,
            "trade_limit": 1,
            "stop_duration": 60,
            "required_profit": -0.05
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 24,
            "trade_limit": 1,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.2
        },
    ]

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if (current_profit > 0.3):
            return 0.05
        elif (current_profit > 0.1):
            return 0.03
        elif (current_profit > 0.06):
            return 0.02
        elif (current_profit > 0.04):
            return 0.01
        elif (current_profit > 0.025):
            return 0.005
        elif (current_profit > 0.018):
            return 0.005

        return 0.15

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if (last_candle is not None):
            if (sell_reason in ['sell_signal']):
                if (last_candle['hma_50']*1.149 > last_candle['ema_100']) and (last_candle['close'] < last_candle['ema_100']*0.951):  # *1.2
                    return False

        # slippage
        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        candle = dataframe.iloc[-1].squeeze()

        slippage = (rate / candle['close']) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                return False

        state[pair] = 0

        return True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '15m') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_1h

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)
        # EMA
        # informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        # informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # # RSI
        # informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # informative_1h['bb_lowerband'] = bollinger['lower']
        # informative_1h['bb_middleband'] = bollinger['mid']
        # informative_1h['bb_upperband'] = bollinger['upper']

        return informative_15m

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=5)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=25)

        """
        ANTIPUMP THING TO TEST
        """
        dataframe['pct_change'] = dataframe['close'].pct_change(periods=8)
        dataframe['pct_change_int'] = ((dataframe['pct_change'] > 0.15).astype(int) | (dataframe['pct_change'] < -0.15).astype(int))

        dataframe['pct_change_short'] = dataframe['close'].pct_change(periods=8)
        dataframe['pct_change_int_short'] = ((dataframe['pct_change_short'] > 0.08).astype(int) | (dataframe['pct_change_short'] < -0.08).astype(int))

        dataframe['ispumping'] = (
         (dataframe['pct_change_int'].rolling(20).sum() >= 0.4)
        ).astype('int')

        dataframe['islongpumping'] = (
         (dataframe['pct_change_int'].rolling(30).sum() >= 0.48)
        ).astype('int')

        dataframe['isshortpumping'] = (
         (dataframe['pct_change_int_short'].rolling(10).sum() >= 0.10)
        ).astype('int')

        dataframe['recentispumping'] = (dataframe['ispumping'].rolling(300).max() > 0) | (dataframe['islongpumping'].rolling(300).max() > 0)# | (dataframe['isshortpumping'].rolling(300).max() > 0)
        """
        END ANTIPUMP
        """

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # informative_1h = self.informative_1h_indicators(dataframe, metadata)
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                # don't buy if there isn't 3% profit to be made
                (dataframe['close_15m'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['rsi'] < 25)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            ((dataframe['close'] > dataframe['sma_9']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi'] > 50) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
             )
            |
            (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe

class NASOSv5HO(NASOSv5_mod1):
    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 8,
        "ewo_high": 4.13,
        "ewo_high_2": 4.477,
        "ewo_low": -19.076,
        "lookback_candles": 27,
        "low_offset": 0.988,
        "low_offset_2": 0.974,
        "profit_threshold": 1.049,
        "rsi_buy": 72,
        "rsi_fast_buy": 40,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 8,
        "high_offset": 1.012,
        "high_offset_2": 1.431,
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.1
    }

    # Stoploss:
    stoploss = -0.1  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.001  # value loaded from strategy
    trailing_stop_positive_offset = 0.03  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

class NASOSv5PD(NASOSv5_mod1):
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dont_buy_conditions = []

        dont_buy_conditions.append(
            (
                # don't buy if there isn't 3% profit to be made
                (dataframe['close_15m'].rolling(self.lookback_candles.value).max()
                 < (dataframe['close'] * self.profit_threshold.value))
            )
        )

        dont_buy_conditions.append(
            (dataframe['recentispumping'] == True)
        )

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewo1')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] > self.ewo_high_2.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['rsi'] < 25)
            ),
            ['buy', 'buy_tag']] = (1, 'ewo2')

        dataframe.loc[
            (
                (dataframe['rsi_fast'] < self.rsi_fast_buy.value) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
            ),
            ['buy', 'buy_tag']] = (1, 'ewolow')

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'buy'] = 0

        return dataframe

class NASOSv5SL(NASOSv5_mod1):
    sell_params = {
        "pHSL": -0.178,
        "pPF_1": 0.019,
        "pPF_2": 0.065,
        "pSL_1": 0.019,
        "pSL_2": 0.062,
        "base_nb_candles_sell": 12,
        "high_offset": 1.01,
        "high_offset_2": 1.142,
    }

    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)


    trailing_stop = False
    use_custom_stoploss = True

    ## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
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


class TrailingBuyStrat(NASOSv5_mod1):
    # if process_only_new_candles = True, then you need to use 1m timeframe (and normal strat timeframe as informative)

    trailing_buy_order_enabled = True
    trailing_buy_offset = 0.005
    process_only_new_candles = True

    custom_info = dict() # custom_info should be a dict

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        tag = super(TrailingBuyStrat, self).custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if tag:
            self.custom_info[pair]['trailing_buy'] = {
                'trailing_buy_order_started': False,
                'trailing_buy_order_uplimit': 0,
                'start_trailing_price': 0,
                'buy_tag': None
            }
            logger.info(f'STOP trailing buy for {pair} because of {tag}')
        return tag

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super(TrailingBuyStrat, self).populate_indicators(dataframe, metadata)
        if not metadata["pair"] in self.custom_info:
            self.custom_info[metadata["pair"]] = dict()
        if not 'trailing_buy' in self.custom_info[metadata['pair']]:
            self.custom_info[metadata["pair"]]['trailing_buy'] = {
                'trailing_buy_order_started': False,
                'trailing_buy_order_uplimit': 0,
                'start_trailing_price': 0,
                'buy_tag': None
            }
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        val = super(TrailingBuyStrat, self).confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs)
        self.custom_info[pair]['trailing_buy']['trailing_buy_order_started'] = False
        self.custom_info[pair]['trailing_buy']['trailing_buy_order_uplimit'] = 0
        self.custom_info[pair]['trailing_buy']['start_trailing_price'] = 0
        self.custom_info[pair]['trailing_buy']['buy_tag'] = None
        return val

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        def get_local_min(x):
            win = dataframe.loc[:, 'barssince_last_buy'].iloc[x.shape[0] - 1].astype('int')
            win = max(win, 0)
            return pd.Series(x).rolling(window=win).min().iloc[-1]

        dataframe = super(TrailingBuyStrat, self).populate_buy_trend(dataframe, metadata)
        dataframe = dataframe.rename(columns={"buy": "pre_buy"})

        if self.trailing_buy_order_enabled and self.config['runmode'].value in ('live', 'dry_run'):  # trailing live dry ticker, 1m
            last_candle = dataframe.iloc[-1].squeeze()
            if not self.process_only_new_candles:
                current_price = self.get_current_price(metadata["pair"])
            else:
                current_price = last_candle['close']
            dataframe['buy'] = 0
            if not self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started'] and last_candle['pre_buy'] == 1:
                self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started'] = True
                self.custom_info[metadata["pair"]]['trailing_buy']['start_trailing_price'] = last_candle['close']
                self.custom_info[metadata["pair"]]['trailing_buy']['buy_tag'] = last_candle['buy_tag']
                self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = last_candle[f'close']
                logger.info(f'start trailing buy for {metadata["pair"]} at {last_candle["close"]}')
            elif self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started']:
                if current_price < self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit']:
                    self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = min(current_price * (1 + self.trailing_buy_offset), self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'])
                    logger.info(f'update trailing buy for {metadata["pair"]} at {self.custom_info[metadata["pair"]]["trailing_buy"]["trailing_buy_order_uplimit"]}')
                elif current_price < self.custom_info[metadata["pair"]]['trailing_buy']['start_trailing_price']:
                    dataframe.iloc[-1, dataframe.columns.get_loc('buy')] = 1
                    ratio = "%.2f" % ((current_price / self.custom_info[metadata['pair']]['trailing_buy']['start_trailing_price']) * 100)
                    dataframe.iloc[-1, dataframe.columns.get_loc('buy_tag')] = f"{self.custom_info[metadata['pair']]['trailing_buy']['buy_tag']} ({ratio} %)"
                    # stop trailing when buy signal ! prevent from buyin much higher price when slot is free
                    self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_started'] = False
                    self.custom_info[metadata["pair"]]['trailing_buy']['trailing_buy_order_uplimit'] = 0
                    self.custom_info[metadata["pair"]]['trailing_buy']['start_trailing_price'] = None
                    self.custom_info[metadata["pair"]]['trailing_buy']['buy_tag'] = None
                else:
                    logger.info(f'price to high for {metadata["pair"]} at {current_price} vs {self.custom_info[metadata["pair"]]["trailing_buy"]["trailing_buy_order_uplimit"]}')
        elif self.trailing_buy_order_enabled:
            # FOR BACKTEST
            # PROBABLY STILL NOT WORKING
            dataframe.loc[
                (dataframe['pre_buy'] == 1) &
                (dataframe['pre_buy'].shift() == 0)
            , 'pre_buy_switch'] = 1
            dataframe['pre_buy_switch'] = dataframe['pre_buy_switch'].fillna(0)

            dataframe['barssince_last_buy'] = dataframe['pre_buy_switch'].groupby(dataframe['pre_buy_switch'].cumsum()).cumcount()

            # Create integer positions of each row
            idx_positions = np.arange(len(dataframe))
            # "shift" those integer positions by the amount in shift col
            shifted_idx_positions = idx_positions - dataframe["barssince_last_buy"]
            # get the label based index from our DatetimeIndex
            shifted_loc_index = dataframe.index[shifted_idx_positions]
            # Retrieve the "shifted" values and assign them as a new column
            dataframe["close_5m_last_buy"] = dataframe.loc[shifted_loc_index, "close_5m"].values

            dataframe.loc[:, 'close_lower'] = dataframe.loc[:, 'close'].expanding().apply(get_local_min)
            dataframe['close_lower'] = np.where(dataframe['close_lower'].isna() == True, dataframe['close'], dataframe['close_lower'])
            dataframe['close_lower_offset'] = dataframe['close_lower'] * (1 + self.trailing_buy_offset)
            dataframe['trailing_buy_order_uplimit'] = np.where(dataframe['barssince_last_buy'] < 20, pd.DataFrame([dataframe['close_5m_last_buy'], dataframe['close_lower_offset']]).min(), np.nan)

            dataframe.loc[
                (dataframe['barssince_last_buy'] < 20) & # must buy within last 20 candles after signal
                (dataframe['close'] > dataframe['trailing_buy_order_uplimit'])
            , 'trailing_buy'] = 1

            dataframe['trailing_buy_count'] = dataframe['trailing_buy'].rolling(20).sum()

            dataframe.log[
                (dataframe['trailing_buy'] == 1) &
                (dataframe['trailing_buy_count'] == 1)
            , 'buy'] = 1
        else: # No but trailing
            dataframe.loc[
                (dataframe['pre_buy'] == 1)
            , 'buy'] = 1
        return dataframe

    def get_current_price(self, pair: str) -> float:
        ticker = self.dp.ticker(pair)
        current_price = ticker['last']
        return current_price
