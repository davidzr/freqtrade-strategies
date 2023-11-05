# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class bbrsi1_strategy (IStrategy):

   # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

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

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger bands stds 2
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband_2'] = bollinger['lower']
        dataframe['bb_middleband_2'] = bollinger['mid']
        dataframe['bb_upperband_2'] = bollinger['upper']
        
        # Bollinger bands stds 3
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband_3'] = bollinger['lower']
        dataframe['bb_middleband_3'] = bollinger['mid']
        dataframe['bb_upperband_3'] = bollinger['upper']
        
        # Bollinger bands stds 4
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband_4'] = bollinger['lower']
        dataframe['bb_middleband_4'] = bollinger['mid']
        dataframe['bb_upperband_4'] = bollinger['upper']

        return dataframe
        
    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True),
    buy_rsi_enabled = CategoricalParameter([True, False], default=True, space="buy", optimize=True),
    buy_trigger = CategoricalParameter(["bb_lowerband_2", "bb_lowerband_3", "bb_lowerband_4",], default="bb_lowerband_2", space="buy", 	optimize=True),
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True),
    sell_rsi_enabled = CategoricalParameter([True, False], default=True, space="sell"),
    sell_trigger = CategoricalParameter(["bb_middleband_2", "bb_middleband_3", "bb_middleband_4", "bb_upperband_2", "bb_upperband_3", 	"bb_upperband_4"], default="bb_upperband_2", space="sell")


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] < 30) &
                    (dataframe['close'] < dataframe['bb_lowerband_2']) &
                    (dataframe['volume'] > 0)

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 70) &
                    (dataframe['volume'] > 0)

            ),
            'sell'] = 1
        return dataframe

