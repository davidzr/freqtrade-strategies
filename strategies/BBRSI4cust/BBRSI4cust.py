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


# This class is a sample. Feel free to customize it.
class BBRSI4cust(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        # "60": 0.01,
        # "30": 0.02,
        "0": 0.003
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    # buy_rsi = IntParameter(low=25, high=35, default=35, space='buy', optimize=True, load=True)
    buy_bb = IntParameter(low=1, high=4, default=1, space='buy', optimize=True, load=True)
    buy_di = IntParameter(low=10, high=20, default=20, space='buy', optimize=True, load=True)

    sell_bb = IntParameter(low=1, high=4, default=1, space='sell', optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'bb_lowerband': {'color': 'blue'},
            'bb_middleband': {'color': 'orange'},            
            'bb_upperband': {'color': 'blue'},
        },
        'subplots': {
            "DI": {
                'plus_di': {'color': 'green'},
                'di_overbought': {'color': 'black'},
                
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Momentum Indicators
        # ------------------------------------

        # Plus Directional Indicator / Movement
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['di_overbought'] = 20
        

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=self.buy_bb.value)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=self.sell_bb.value)
        dataframe['bb_lowerband1'] = bollinger1['lower']
        dataframe['bb_middleband1'] = bollinger1['mid']
        dataframe['bb_upperband1'] = bollinger1['upper']


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                (dataframe['plus_di'] > self.buy_di.value) &  
                (qtpylib.crossed_below(dataframe['low'], dataframe['bb_lowerband'])) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(dataframe['high'], dataframe['bb_middleband1'])) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),

            'exit_long'] = 1

        return dataframe


    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        """
        Sell only when matching some criteria other than those used to generate the sell signal
        :return: str sell_reason, if any, otherwise None
        """
        # get dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        
        # get the current candle
        current_candle = dataframe.iloc[-1].squeeze()
        
        # if (qtpylib.crossed_above(current_candle['high'], dataframe['bb_middleband1'])) == True:

        if (qtpylib.crossed_above(current_rate, current_candle['bb_middleband1'])):

            return "bb_profit_sell"

        # else, hold
        return None