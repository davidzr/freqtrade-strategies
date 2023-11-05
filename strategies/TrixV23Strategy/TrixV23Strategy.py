# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (merge_informative_pair, 
                                BooleanParameter, 
                                CategoricalParameter, 
                                DecimalParameter,
                                IStrategy, 
                                IntParameter, 
                                informative)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import PairLocks, Trade
from datetime import datetime


# This class is a sample. Feel free to customize it.
class TrixV23Strategy(IStrategy):
    """
    Sources : 
    Cripto Robot : https://www.youtube.com/watch?v=uE04UROWkjs&list=PLpJ7cz_wOtsrqEQpveLc2xKLjOBgy4NfA&index=4
    Github : https://github.com/CryptoRobotFr/TrueStrategy/blob/main/TrixStrategy/Trix_Complete_backtest.ipynb
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.553,
        "423": 0.144,
        "751": 0.059,
        "1342": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.31

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = True
    # trailing_stop_positive = 0.02
    # trailing_stop_positive_offset = 0.9  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    use_custom_stoploss = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    #---------------------------#
    #-- Hyperspace parameters --#
    #---------------------------#

    buy_params = {
        "buy_trix_signal_type": "trigger",
        "buy_trix_src": "low",
        "buy_trix_timeperiod": 8,
        "buy_trix_signal_timeperiod": 19,

        "buy_stoch_rsi_enabled": True,
        "buy_rsi_timeperiod": 14,
        "buy_stoch_rsi_timeperiod": 14,
        "buy_stoch_rsi": 0.901,

        "buy_ema_enabled": True,
        "buy_ema_src": "open",
        "buy_ema_timeperiod": 10,
        "buy_ema_multiplier": 0.85,

        "buy_btc_ema_enabled": True,
        "buy_btc_ema_multiplier": 0.996,
        "buy_btc_ema_timeperiod": 184,
    }

    sell_params = {
        "sell_trix_signal_type": "trailing",
        "sell_trix_src": "high",
        "sell_trix_timeperiod": 10,
        "sell_trix_signal_timeperiod": 19,

        "sell_stoch_rsi_enabled": True,
        "sell_rsi_timeperiod": 14,
        "sell_stoch_rsi_timeperiod": 14,
        "sell_stoch_rsi": 0.183,

        "sell_atr_enabled": True,
        "sell_atr_timeperiod": 30,
        "sell_atr_multiplier": 4.99,
    }

    #------------------------------#
    #-- Hyperoptables parameters --#
    #------------------------------#
    
    # buy

    buy_trix_signal_type = CategoricalParameter(['trailing', 'trigger'], default='trigger', space="buy", optimize=False, load=True)
    buy_trix_src = CategoricalParameter(['open', 'high', 'low', 'close'], default='close', space="buy", optimize=False, load=True)
    buy_trix_timeperiod = IntParameter(5, 25, default=9, space="buy", optimize=False, load=True)
    buy_trix_signal_timeperiod = IntParameter(5, 25, default=21, space="buy", optimize=False, load=True)

    buy_stoch_rsi_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    buy_rsi_timeperiod = IntParameter(5, 25, default=14, space="buy", optimize=False, load=True)
    buy_stoch_rsi = DecimalParameter(0.6, 0.99, decimals=3, default=0.987, space="buy", optimize=False, load=True)
    buy_stoch_rsi_timeperiod = IntParameter(5, 25, default=14, space="buy", optimize=False, load=True)

    buy_ema_enabled = BooleanParameter(default=False, space="buy", optimize=False, load=True)
    buy_ema_timeperiod = IntParameter(9, 100, default=21, space="buy", optimize=False, load=True)
    buy_ema_multiplier = DecimalParameter(0.8, 1.2, decimals=2, default=1.00, space="buy", optimize=False, load=True)
    buy_ema_src = CategoricalParameter(['open', 'high', 'low', 'close'], default='close', space="buy", optimize=False, load=True)

    buy_btc_ema_enabled = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    buy_btc_ema_timeperiod = IntParameter(150, 250, default=200, space="buy", optimize=True, load=True)
    buy_btc_ema_multiplier = DecimalParameter(0.8, 1.0, decimals=3, default=0.97, space="buy", optimize=True, load=True)

    # sell

    sell_trix_signal_type = CategoricalParameter(['trailing', 'trigger'], default='trailing', space="sell", optimize=False, load=True)
    sell_trix_src = CategoricalParameter(['open', 'high', 'low', 'close'], default='close', space="sell", optimize=False, load=True)
    sell_trix_timeperiod = IntParameter(5, 25, default=9, space="sell", optimize=False, load=True)
    sell_trix_signal_timeperiod = IntParameter(5, 25, default=21, space="sell", optimize=False, load=True)

    sell_stoch_rsi_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    sell_rsi_timeperiod = IntParameter(5, 25, default=14, space="sell", optimize=False, load=True)
    sell_stoch_rsi = DecimalParameter(0.01, 0.4, decimals=3, default=0.048, space="sell", optimize=False, load=True)
    sell_stoch_rsi_timeperiod = IntParameter(5, 25, default=14, space="sell", optimize=False, load=True)

    sell_atr_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    sell_atr_timeperiod = IntParameter(9, 30, default=14, space="sell", optimize=False, load=True)
    sell_atr_multiplier = DecimalParameter(0.7, 9.0, decimals=3, default=4.0, space="sell", optimize=False, load=True)

    plot_config = {
        'main_plot': {
            'trix_b_8': {'color': 'blue'},
            'trix_s_10': {'color': 'orange'},
            'ema_b_signal': {'color': 'red'},
            'btc_usdt_close_1h': {'color': 'purple'},
            'btc_usdt_ema_184_1h': {'color': 'yellow'},
        },
         'subplots': {
            "TRIX BUY": {
                'trix_b_pct': {'color': 'blue'},
                'trix_b_signal_19': {'color': 'orange'},
            },
            "TRIX SELL": {
                'trix_s_pct': {'color': 'blue'},
                'trix_s_signal_19': {'color': 'orange'},
            },
            "STOCH RSI": {
                'b_stoch_rsi': {'color': 'blue'},
                's_stoch_rsi': {'color': 'orange'},
            },
        }
    }

    @informative('1h', 'BTC/{stake}')
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #---------#
        #-- BTC --#
        #---------#

        for val in self.buy_btc_ema_timeperiod.range:
            dataframe[f'ema_{val}'] = ta.EMA(dataframe, timeperiod=val)

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:

        #------------------------#
        #-- ATR based stoploss --#
        #------------------------#

        if self.sell_atr_enabled.value == False:
            return 1

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        stoploss_price = last_candle['low'] - last_candle[f'atr_{self.sell_atr_timeperiod.value}'] * self.sell_atr_multiplier.value

        if stoploss_price < current_rate:
            return (stoploss_price / current_rate) - 1

        # return maximum stoploss value, keeping current stoploss price unchanged
        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #----------------#
        #-- Indicators --#
        #----------------#

        # Trix Indicator
        for val in self.buy_trix_timeperiod.range:
            dataframe[f'trix_b_{val}'] = ta.EMA(ta.EMA(ta.EMA(dataframe[self.buy_trix_src.value], timeperiod=val), timeperiod=val), timeperiod=val)
        dataframe['trix_b_pct'] = dataframe[f'trix_b_{self.buy_trix_timeperiod.value}'].pct_change() * 100
        for val in self.buy_trix_signal_timeperiod.range:
            dataframe[f'trix_b_signal_{val}'] = ta.SMA(dataframe['trix_b_pct'], timeperiod=val)

        for val in self.sell_trix_timeperiod.range:
            dataframe[f'trix_s_{val}'] = ta.EMA(ta.EMA(ta.EMA(dataframe[self.sell_trix_src.value], timeperiod=val), timeperiod=val), timeperiod=val)
        dataframe['trix_s_pct'] = dataframe[f'trix_s_{self.sell_trix_timeperiod.value}'].pct_change() * 100
        for val in self.sell_trix_signal_timeperiod.range:
            dataframe[f'trix_s_signal_{val}'] = ta.SMA(dataframe['trix_s_pct'], timeperiod=val)

        # Stochastic RSI
        for val in self.buy_rsi_timeperiod.range:
            dataframe['b_rsi'] = ta.RSI(dataframe, timeperiod=val)
        for val in self.buy_stoch_rsi_timeperiod.range:
            dataframe['b_stoch_rsi'] = (dataframe['b_rsi'] - dataframe['b_rsi'].rolling(val).min()) / (dataframe['b_rsi'].rolling(val).max() - dataframe['b_rsi'].rolling(val).min())

        for val in self.sell_rsi_timeperiod.range:
            dataframe['s_rsi'] = ta.RSI(dataframe, timeperiod=val)
        for val in self.sell_stoch_rsi_timeperiod.range:
            dataframe['s_stoch_rsi'] = (dataframe['s_rsi'] - dataframe['s_rsi'].rolling(val).min()) / (dataframe['s_rsi'].rolling(val).max() - dataframe['s_rsi'].rolling(val).min())

        # EMA
        for val in self.buy_ema_timeperiod.range:
            dataframe[f'ema_b_{val}'] = ta.EMA(dataframe[self.buy_ema_src.value], timeperiod=val)
        dataframe['ema_b_signal'] = dataframe[f'ema_b_{self.buy_ema_timeperiod.value}'] * self.buy_ema_multiplier.value

        # ATR
        for val in self.sell_atr_timeperiod.range:
            dataframe[f'atr_{val}'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=val)
        dataframe['stoploss_price'] = dataframe['low'] - dataframe[f'atr_{self.sell_atr_timeperiod.value}'] * self.sell_atr_multiplier.value

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        stake = self.config['stake_currency'].lower()

        #-----------------------#
        #-- Guards and trends --#
        #-----------------------#

        # For backtesting & Hyperopt
        conditions.append(dataframe['volume'] > 0)

        # Can't send a buy signal at the same time as a sell signal
        conditions.append(dataframe['trix_s_pct'] > dataframe[f'trix_s_signal_{self.sell_trix_signal_timeperiod.value}'])

        # If BTC is not going well, don't buy
        if self.buy_btc_ema_enabled.value:
            conditions.append(dataframe[f'btc_{stake}_close_1h'] > (dataframe[f'btc_{stake}_ema_{self.buy_btc_ema_timeperiod.value}_1h'] * self.buy_btc_ema_multiplier.value))
        
        # Stoch RSI
        if self.buy_stoch_rsi_enabled.value:
            conditions.append(dataframe['b_stoch_rsi'] < self.buy_stoch_rsi.value)

        # Trend check
        if self.buy_ema_enabled.value:
            conditions.append(dataframe['close'] > dataframe['ema_b_signal'])

        # Probably less efficient than trigger mode
        if self.buy_trix_signal_type.value == 'trailing':
            conditions.append(dataframe['trix_b_pct'] > dataframe[f'trix_b_signal_{self.buy_trix_signal_timeperiod.value}'])

        #--------------#
        #-- Triggers --#
        #--------------#

        # Main trigger : trix indicator
        if self.buy_trix_signal_type.value == 'trigger':
            conditions.append(qtpylib.crossed_above(dataframe['trix_b_pct'], dataframe[f'trix_b_signal_{self.buy_trix_signal_timeperiod.value}']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []

        #-----------------------#
        #-- Guards and trends --#
        #-----------------------#

        # For backtesting & Hyperopt
        conditions.append(dataframe['volume'] > 0)

        # Stoch RSI
        if self.sell_stoch_rsi_enabled.value:
            conditions.append(dataframe['s_stoch_rsi'] > self.sell_stoch_rsi.value)

        # Main indicator : Trix
        if self.sell_trix_signal_type.value == 'trailing':
            conditions.append(dataframe['trix_s_pct'] < dataframe[f'trix_s_signal_{self.sell_trix_signal_timeperiod.value}'])
        
        #--------------#
        #-- Triggers --#
        #--------------#

        # Main indicator. We probably want trailing mode
        if self.sell_trix_signal_type.value == 'trigger':
            conditions.append(qtpylib.crossed_below(dataframe['trix_s_pct'], dataframe[f'trix_s_signal_{self.sell_trix_signal_timeperiod.value}']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1
        return dataframe