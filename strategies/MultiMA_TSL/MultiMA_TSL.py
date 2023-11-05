import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter, stoploss_from_open)
from pandas import DataFrame
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_prev_date
from technical.indicators import zema

###########################################################################################################
##    MultiMA_TSL, modded by stash86, based on SMAOffsetProtectOptV1 (modded by Perkmeister)             ##
##    Based on @Lamborghini Store's SMAOffsetProtect strat, heavily based on @tirail's original SMAOffset##
##                                                                                                       ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##
##                                                                                                       ##
###########################################################################################################

# I hope you do enough testing before proceeding, either backtesting and/or dry run.
# Any profits and losses are all your responsibility

class MultiMA_TSL(IStrategy):
    INTERFACE_VERSION = 2

    buy_params = {
        "base_nb_candles_buy_ema": 50,
        "low_offset_ema": 1.061,

        "base_nb_candles_buy_zema": 30,
        "low_offset_zema": 0.963,
        "rsi_buy_zema": 50,

        "base_nb_candles_buy_trima": 14,
        "low_offset_trima": 0.963,
        "rsi_buy_trima": 50,

        "buy_roc_max": 45,

        "buy_condition_trima_enable": True,
        "buy_condition_zema_enable": True,
    }

    sell_params = {
        "base_nb_candles_sell": 32,
        "high_offset_ema": 1.002,

        "base_nb_candles_sell_trima": 48,
        "high_offset_trima": 1.085,
    }

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    stoploss = -0.15

    # Multi Offset
    base_nb_candles_sell = IntParameter(5, 80, default=20, space='sell', optimize=False)
    base_nb_candles_sell_trima = IntParameter(5, 80, default=20, space='sell', optimize=False)
    high_offset_trima = DecimalParameter(0.99, 1.1, default=1.012, space='sell', optimize=False)

    base_nb_candles_buy_ema = IntParameter(5, 80, default=20, space='buy', optimize=False)
    low_offset_ema = DecimalParameter(0.9, 1.1, default=0.958, space='buy', optimize=False)
    high_offset_ema = DecimalParameter(0.99, 1.1, default=1.012, space='sell', optimize=False)
    rsi_buy_ema = IntParameter(30, 70, default=61, space='buy', optimize=False)

    base_nb_candles_buy_trima = IntParameter(5, 80, default=20, space='buy', optimize=False)
    low_offset_trima = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=False)
    rsi_buy_trima = IntParameter(30, 70, default=61, space='buy', optimize=False)

    base_nb_candles_buy_zema = IntParameter(5, 80, default=20, space='buy', optimize=False)
    low_offset_zema = DecimalParameter(0.9, 0.99, default=0.958, space='buy', optimize=False)
    rsi_buy_zema = IntParameter(30, 70, default=61, space='buy', optimize=False)

    buy_condition_enable_optimize = True
    buy_condition_trima_enable = BooleanParameter(default=True, space='buy', optimize=buy_condition_enable_optimize)
    buy_condition_zema_enable = BooleanParameter(default=True, space='buy', optimize=buy_condition_enable_optimize)

    # Protection1
    ewo_low = DecimalParameter(-20.0, -8.0, default=-20.0, space='buy', optimize=False)
    ewo_high = DecimalParameter(2.0, 12.0, default=6.0, space='buy', optimize=False)
    fast_ewo = IntParameter(10, 50, default=50, space='buy', optimize=False)
    slow_ewo = IntParameter(100, 200, default=200, space='buy', optimize=False)
    buy_roc_max = DecimalParameter(20, 70, default=55, space='buy', optimize=False)
    buy_peak_max = DecimalParameter(1, 1.1, default=1.03, decimals=3, space='buy', optimize=False)
    buy_rsi_fast = IntParameter(0, 50, default=35, space='buy', optimize=False)

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.018

    use_custom_stoploss = True

    # Protection hyperspace params:
    protection_params = {
        "low_profit_lookback": 60,
        "low_profit_min_req": 0.03,
        "low_profit_stop_duration": 29,

        "cooldown_lookback": 2,  # value loaded from strategy
        "stoploss_lookback": 72,  # value loaded from strategy
        "stoploss_stop_duration": 20,  # value loaded from strategy
    }

    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=False)

    low_profit_lookback = IntParameter(2, 60, default=20, space="protection", optimize=False)
    low_profit_stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=False)
    low_profit_min_req = DecimalParameter(-0.05, 0.05, default=-0.05, space="protection", decimals=2, optimize=False)

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

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    #credit to Perkmeister for this custom stoploss to help the strategy ride a green candle when the sell signal triggered
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if not self.config['runmode'].value in ('backtest', 'hyperopt'):
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if(len(dataframe) >= 1):
                last_candle = dataframe.iloc[-1]
                if((last_candle['sell_copy'] == 1) & (last_candle['buy_copy'] == 0)):
                    sl_new = 0.001

        return sl_new

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EWO
        dataframe['ewo'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)

        dataframe["roc_max"] = dataframe["close"].pct_change(48).rolling(12).max() * 100
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(self.base_nb_candles_buy_ema.value)) *self.low_offset_ema.value
        dataframe['zema_offset_buy'] = zema(dataframe, int(self.base_nb_candles_buy_zema.value)) *self.low_offset_zema.value
        dataframe['trima_offset_buy'] = ta.TRIMA(dataframe, int(self.base_nb_candles_buy_trima.value)) *self.low_offset_trima.value
        
        dataframe.loc[:, 'buy_tag'] = ''
        dataframe.loc[:, 'buy_copy'] = 0
        dataframe.loc[:, 'buy'] = 0

        buy_offset_trima = (
            self.buy_condition_trima_enable.value &
            (dataframe['close'] < dataframe['trima_offset_buy']) &
            (
                (dataframe['ewo'] < self.ewo_low.value)
                |
                (
                    (dataframe['ewo'] > self.ewo_high.value)
                    &
                    (dataframe['rsi'] < self.rsi_buy_trima.value)
                )
            )
        )
        dataframe.loc[buy_offset_trima, 'buy_tag'] += 'trima '
        conditions.append(buy_offset_trima)

        buy_offset_zema = (
            self.buy_condition_zema_enable.value &
            (dataframe['close'] < dataframe['zema_offset_buy']) &
            (
                (dataframe['ewo'] < self.ewo_low.value)
                |
                (
                    (dataframe['ewo'] > self.ewo_high.value)
                    &
                    (dataframe['rsi'] < self.rsi_buy_zema.value)
                )
            )
        )
        dataframe.loc[buy_offset_zema, 'buy_tag'] += 'zema '
        conditions.append(buy_offset_zema)

        add_check = (
            (dataframe['rsi_fast'] < self.buy_rsi_fast.value)
            &
            (dataframe['close'] < dataframe['ema_offset_buy'])
            &
            (dataframe['volume'] > 0)
        )
        
        if conditions:
            dataframe.loc[
                (add_check & reduce(lambda x, y: x | y, conditions)),
                ['buy_copy','buy']
            ]=(1,1)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell_copy'] = 0

        dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(self.base_nb_candles_sell.value)) *self.high_offset_ema.value
        dataframe['trima_offset_sell'] = ta.TRIMA(dataframe, int(self.base_nb_candles_sell_trima.value)) *self.high_offset_trima.value

        conditions = []

        conditions.append(
            (
                (dataframe['close'] > dataframe['ema_offset_sell']) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                (dataframe['close'] > dataframe['trima_offset_sell']) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                ['sell_copy', 'sell']
            ]=(1,1)

        if not self.config['runmode'].value in ('backtest', 'hyperopt'):
            dataframe.loc[:, 'sell'] = 0

        return dataframe


# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif
