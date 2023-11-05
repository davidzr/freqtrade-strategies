# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_open

pd.options.mode.chained_assignment = None  # default='warn'

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class PrawnstarOBV(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Optimal timeframe for the strategy
    timeframe = '1h'

    # ROI table:
    #minimal_roi = {
    #    "0": 0.8
    #}

    minimal_roi = {
        "0": 0.296,
        "179": 0.137,
        "810": 0.025,
        "1024": 0
    }

    # Stoploss:
    stoploss = -0.15

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = False
    use_buy_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Momentum Indicators
        # ------------------------------------
        
        # Momentum
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obvSma'] = ta.SMA(dataframe['obv'], timeperiod=7)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['obv'], dataframe['obvSma'])) &
                (dataframe['rsi'] < 50) |
                ((dataframe['obvSma'] - dataframe['close']) / dataframe['obvSma'] > 0.1) |
                (dataframe['obv'] > dataframe['obv'].shift(1)) &
                (dataframe['obvSma'] > dataframe['obvSma'].shift(5)) &
                (dataframe['rsi'] < 50)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
            ),
            'sell'] = 1

        return dataframe